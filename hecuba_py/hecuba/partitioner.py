import os
import time
import uuid
from bisect import bisect_right
from collections import defaultdict, deque

from hecuba import config, log

_select_istorage_meta = config.session.prepare("SELECT * FROM hecuba.istorage WHERE storage_id = ?")
_size_estimates = config.session.prepare(("SELECT mean_partition_size, partitions_count "
                                          "FROM system.size_estimates WHERE keyspace_name=? and table_name=?"))

_max_token = int(((2 ** 63) - 1))  # type: int
_min_token = int(-2 ** 63)  # type: int


def partitioner_split(father):
    if hasattr(config, "partition_strategy"):
        return Partitioner(father, config.partition_strategy).split()
    else:
        return Partitioner(father, "SIMPLE").split()


class Partitioner:

    def __init__(self, father, strategy):
        self._father = father
        self._strategy = strategy
        self._rebuild_token_ring(self._father._ksp, self._father._build_args.tokens)
        if strategy == "DYNAMIC":
            self._setup_dynamic_structures()

    def _rebuild_token_ring(self, ksp, tokens_ranges):
        tm = config.cluster.metadata.token_map
        tmap = tm.tokens_to_hosts_by_ks.get(ksp, None)
        from cassandra.metadata import Murmur3Token
        tokens_murmur3 = map(lambda a: (Murmur3Token(a[0]), a[1]), tokens_ranges)
        if not tmap:
            tm.rebuild_keyspace(ksp, build_if_absent=True)
            tmap = tm.tokens_to_hosts_by_ks[ksp]

        self._tokens_per_node = defaultdict(list)
        for tmumur, t_to in tokens_murmur3:
            point = bisect_right(tm.ring, tmumur)
            if point == len(tm.ring):
                self._tokens_per_node[tmap[tm.ring[0]][0]].append((tmumur.value, t_to))
            else:
                self._tokens_per_node[tmap[tm.ring[point]][0]].append((tmumur.value, t_to))

        self._nodes_number = len(self._tokens_per_node)

    def _setup_dynamic_structures(self):
        try:
            config.session.execute("""CREATE TABLE IF NOT EXISTS hecuba.partitioning(
                                        partitioning_uuid uuid,
                                        storage_id uuid,
                                        number_of_partitions int,
                                        start_time double,
                                        end_time double,
                                        PRIMARY KEY (storage_id))
                                        WITH default_time_to_live = 86400""")
        except Exception as ex:
            print("Could not create table hecuba.partitioning.")
            raise ex

        self._prepared_store_id = \
            config.session.prepare("""INSERT INTO hecuba.partitioning
                                      (partitioning_uuid, storage_id, number_of_partitions)
                                      VALUES (?, ?, ?)""")
        self._partitioning_uuid = uuid.uuid4()
        self._partitions_time = defaultdict(list)
        self._partitions_nodes = dict()
        self._idle_cassandra_nodes = deque()
        self._partitions_size = dict()
        self._best_granularity = None

        self._select_partitions_times = \
            config.session.prepare("""SELECT storage_id, number_of_partitions, start_time, end_time
                                      FROM hecuba.partitioning
                                      WHERE partitioning_uuid = ? ALLOW FILTERING""")

        try:
            self._nodes_number = len(os.environ["PYCOMPSS_NODES"].split(",")) - 1
        except KeyError:
            self._nodes_number = int(os.environ["NODES_NUMBER"]) - 1  # master and worker

        self._n_idle_nodes = self._nodes_number
        self._initial_send = self._nodes_number
        # generate list of basic number of partitions
        partitions = [(int(2 ** (x / 2)) // len(self._tokens_per_node)) for x in range(10, 21)]
        # as many basic number of partitions as the number of nodes
        # 11 basic number of partitions, repeating them when more than 11 nodes
        self._basic_partitions = (partitions * (self._nodes_number // len(partitions) + 1))[0:self._nodes_number]

    def _tokens_partitions(self, ksp, table, token_range_size, target_token_range_size):
        """
        Method that calculates the new token partitions for a given object
        Returns:
            a tuple (node, partition) every time it's called
        """
        partitions_per_node = self._compute_partitions_per_node(ksp, table, token_range_size, target_token_range_size)

        if self._strategy == "DYNAMIC":
            for partition_tokens in self._dynamic_tokens_partitions(partitions_per_node):
                yield partition_tokens
        else:
            for final_tokens in self._send_final_tasks(partitions_per_node):
                yield final_tokens

    def _compute_partitions_per_node(self, ksp, table, token_range_size, target_token_range_size):
        """
        Compute all the partitions per node. If the strategy is simple partitioning, each node will have
        (config.splits_per_node (default 32) * self._nodes_number) partitions. If the strategy is dynamic partitioning,
        each node will have 1024 partitions, because if there is only one Cassandra node this will be the minimum
        granularity. If there are more nodes, some partitions will be grouped.
        Returns:
            a dictionary with hosts as keys and partitions of tokens as values
        """
        step_size = _max_token // (config.splits_per_node * self._nodes_number)

        if token_range_size:
            step_size = token_range_size
        elif target_token_range_size:
            res = config.session.execute(_size_estimates, [ksp, table])
            if res:
                one = res.one()
            else:
                one = 0
            if one:
                (mean_p_size, p_count) = one
                estimated_size = mean_p_size * p_count
                if estimated_size > 0:
                    step_size = _max_token // (
                        max(estimated_size / target_token_range_size,
                            config.splits_per_node * self._nodes_number)
                    )
        if self._strategy == "DYNAMIC":
            # 1024 because it is the maximum number of splits per node, in the case of only one Cassandra node
            step_size = _max_token // 1024

        partitions_per_node = defaultdict(list)
        for node, tokens_in_node in self._tokens_per_node.items():
            for fraction, to in tokens_in_node:
                while fraction < to - step_size:
                    partitions_per_node[node].append((fraction, fraction + step_size))
                    fraction += step_size
                partitions_per_node[node].append((fraction, to))
        return partitions_per_node

    def _dynamic_tokens_partitions(self, partitions_per_node):
        """
        Main loop of the dynamic partitioning strategy. There are 3 stages:
        Sending of initial tasks: it returns as much initial tasks as pycompss nodes
        Sending of intermediate tasks: until a best granularity is chosen, it return partitions using the granularity
        with the best performance
        Sending of final tasks: when a best granularity is chosen, it returns all the remaining partitions using the
        best granularity
        Returns:
            a tuple (node, partition) every time it's called
        """

        while self._initial_send > 0:
            for initial_tokens in self._send_initial_tasks(partitions_per_node):
                yield initial_tokens

        while self._best_granularity is None:
            for intermediate_tokens in self._send_intermediate_tasks(partitions_per_node):
                yield intermediate_tokens
            if sum([len(partitions) for partitions in partitions_per_node.values()]) == 0:
                self._best_granularity = config.splits_per_node
                break
        else:
            for final_tokens in self._send_final_tasks(partitions_per_node):
                yield final_tokens

    def _send_initial_tasks(self, partitions_per_node):
        for node in self._tokens_per_node.keys():
            config.splits_per_node = self._basic_partitions[self._initial_send * -1]
            group_size = max(len(partitions_per_node[node]) // config.splits_per_node, 1)
            if config.splits_per_node not in self._partitions_size:
                self._partitions_size[config.splits_per_node] = group_size

            yield node, partitions_per_node[node][0:group_size]
            del partitions_per_node[node][0:group_size]

            self._partitions_time[config.splits_per_node] = []
            self._initial_send -= 1
            if self._initial_send == 0:
                break

    def _send_intermediate_tasks(self, partitions_per_node):
        self._update_partitions_time()
        while not self._all_tasks_finished():
            if self._n_idle_nodes > 0:
                # if there is an idle node, send a new task without choosing the best granularity
                config.splits_per_node, set_best = self._best_time_per_token()
                if [] not in self._partitions_time.values() and set_best:
                    self._best_granularity = config.splits_per_node
                break
            time.sleep(1)
            self._update_partitions_time()
        else:
            self._best_granularity, _ = self._best_time_per_token()
            config.splits_per_node = self._best_granularity

        node = self._idle_cassandra_nodes.popleft()
        group_size = max(len(partitions_per_node[node]) // config.splits_per_node, 1)
        if config.splits_per_node not in self._partitions_size:
            self._partitions_size[config.splits_per_node] = group_size

        yield node, partitions_per_node[node][0:group_size]
        del partitions_per_node[node][0:group_size]

    def _send_final_tasks(self, partitions_per_node):
        for partition in partitions_per_node.values():
            group_size = max(len(partition) // config.splits_per_node, 1)
            for i in range(0, len(partition), group_size):
                yield -1, partition[i:i + group_size]

    def _all_tasks_finished(self):
        """
        Checks that there is at least one end_time set for all the granularities
        """
        if [] in self._partitions_time.values():
            return False

        for _, partition_times in self._partitions_time.items():
            if not any(times["end_time"] for times in partition_times):
                return False
        return True

    @staticmethod
    def _set_best_granularity(best, unfinished):
        for _, time_per_token in unfinished.items():
            if time_per_token < best:
                return False
        return True

    def _best_time_per_token(self):
        """
        The time is not a good measure, because the smaller tasks will be the shortest.
        We use a time / tokens proportion
        """
        times_per_token = dict()
        unfinished_tasks = dict()
        actual_time = time.time()

        for splits_per_node, partition_times in self._partitions_time.items():
            if len(partition_times) > 0:
                group_size = self._partitions_size[splits_per_node]
                partition_time = 0.0
                if not any(times["end_time"] for times in partition_times):
                    """
                    If there isn't at least one end_time set for this granularity, takes the actual time as the
                    finishing time. If there is already a granularity with better performance, it is selected as the
                    best granularity.
                    A granularity with this condition cannot be set as the best granularity.
                    """
                    for t in partition_times:
                        partition_time += actual_time - t["start_time"]

                    partition_time = partition_time / float(len(partition_times))
                    try:
                        unfinished_tasks[splits_per_node] = partition_time / group_size
                    except ZeroDivisionError:
                        pass
                else:
                    # at least one task finished
                    for t in partition_times:
                        if t["end_time"] is not None:
                            partition_time += t["end_time"] - t["start_time"]

                    partition_time = partition_time / float(len(partition_times))
                    if partition_time >= 2.0:
                        # to avoid having too much overhead, granularities lasting less than two seconds are discarded
                        try:
                            times_per_token[splits_per_node] = partition_time / group_size
                        except ZeroDivisionError:
                            pass

        sorted_times = sorted(times_per_token.items(), key=lambda item: item[1])

        if len(sorted_times) > 0:
            best_granularity, best_time = sorted_times[0]
            set_best = self._set_best_granularity(best_time, unfinished_tasks)
        else:
            # if no task lasted at least two seconds, pick the biggest granularity
            best_granularity = min(set(self._partitions_time.keys()) - set(unfinished_tasks.keys()))
            set_best = False

        return best_granularity, set_best

    def _update_partitions_time(self):
        partitions_times = config.session.execute(self._select_partitions_times, [self._partitioning_uuid])

        for storage_id, partitions, start, end in partitions_times:
            if start is not None:
                for i, times in enumerate(self._partitions_time[partitions]):
                    if start == times["start_time"]:
                        if end is not None and times["end_time"] != end:
                            self._partitions_time[partitions][i]["end_time"] = end
                            self._n_idle_nodes += 1
                            self._idle_cassandra_nodes.append(self._partitions_nodes[storage_id])
                        break
                else:
                    total_time = {"start_time": start, "end_time": end}
                    self._partitions_time[partitions].append(total_time)
                    if end is not None:
                        self._n_idle_nodes += 1
                        self._idle_cassandra_nodes.append(self._partitions_nodes[storage_id])

    def split(self):
        """
        Method used to divide an object into sub-objects.
        Returns:
            a subobject everytime is called
        """
        st = time.time()

        for node, token_split in self._tokens_partitions(self._father._ksp, self._father._table,
                                                         config.token_range_size,
                                                         config.target_token_range_size):
            storage_id = uuid.uuid4()
            log.debug('assigning to %s %d  tokens', str(storage_id), len(token_split))
            new_args = self._father._build_args._replace(tokens=token_split, storage_id=storage_id)
            args_dict = new_args._asdict()
            args_dict["built_remotely"] = False
            if self._strategy == "DYNAMIC":
                config.session.execute(self._prepared_store_id,
                                       [self._partitioning_uuid, storage_id, config.splits_per_node])
                self._n_idle_nodes -= 1
                self._partitions_nodes[storage_id] = node
            yield self._father.__class__.build_remotely(args_dict)

        log.debug('completed split of %s in %f', self.__class__.__name__, time.time() - st)
