#include "ArrayDataStore.h"


ArrayDataStore::ArrayDataStore(const char *table, const char *keyspace, CassSession *session,
                               std::map<std::string, std::string> &config) {

    std::vector<std::map<std::string, std::string> > keys_names = {{{"name", "storage_id"}},
                                                                   {{"name", "cluster_id"}},
                                                                   {{"name", "block_id"}}};

    std::vector<std::map<std::string, std::string> > columns_names = {{{"name", "payload"}}};


    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
    this->cache = new CacheTable(table_meta, session, config);

    std::vector<std::map<std::string, std::string>> read_keys_names(keys_names.begin(), (keys_names.end() - 1));
    std::vector<std::map<std::string, std::string>> read_columns_names = columns_names;
    read_columns_names.insert(read_columns_names.begin(), keys_names.back());

    table_meta = new TableMetadata(table, keyspace, read_keys_names, read_columns_names, session);

    this->read_cache = new CacheTable(table_meta, session, config);
}


ArrayDataStore::~ArrayDataStore() {
    delete (this->cache);
    delete (this->read_cache);
};


/***
 * Write a complete numpy ndarray by using the partitioning mechanism defined in the metadata
 * @param storage_id identifying the numpy ndarray
 * @param np_metas ndarray characteristics
 * @param numpy to be saved into storage
 */
void ArrayDataStore::store(const uint64_t *storage_id, ArrayMetadata &metadata, void *data) const {

    SpaceFillingCurve::PartitionGenerator *partitions_it = this->partitioner.make_partitions_generator(metadata, data);

    char *keys = nullptr;
    void *values = nullptr;
    uint32_t offset = 0, keys_size = sizeof(uint64_t *) + sizeof(int32_t) * 2;
    uint64_t *c_uuid = nullptr;
    uint32_t half_int = 0;//(uint32_t)-1 >> (sizeof(uint32_t)*CHAR_BIT/2); //TODO be done properly
    int32_t cluster_id, block_id;
    while (!partitions_it->isDone()) {
        Partition part = partitions_it->getNextPartition();
        keys = (char *) malloc(keys_size);
        //UUID
        c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);//new uint64_t[2];
        c_uuid[0] = *storage_id;
        c_uuid[1] = *(storage_id + 1);
        // [0] = storage_id.time_and_version;
        // [1] = storage_id.clock_seq_and_node;
        memcpy(keys, &c_uuid, sizeof(uint64_t *));
        offset = sizeof(uint64_t *);
        //Cluster id
        cluster_id = part.cluster_id - half_int;
        memcpy(keys + offset, &cluster_id, sizeof(int32_t));
        offset += sizeof(int32_t);
        //Block id
        block_id = part.block_id - half_int;
        memcpy(keys + offset, &block_id, sizeof(int32_t));
        //COPY VALUES

        values = (char *) malloc(sizeof(char *));
        memcpy(values, &part.data, sizeof(char *));
        //FINALLY WE WRITE THE DATA
        cache->put_crow(keys, values);
    }
    //this->partitioner.serialize_metas();
    delete (partitions_it);

    // No need to flush the elements because the metadata are written after the data thanks to the queue
}


/***
 * Reads a numpy ndarray by fetching the clusters indipendently
 * @param storage_id of the array to retrieve
 * @return Numpy ndarray as a Python object
 */
void *ArrayDataStore::read(const uint64_t *storage_id, ArrayMetadata &metadata) const {

    std::shared_ptr<const std::vector<ColumnMeta> > keys_metas = read_cache->get_metadata()->get_keys();
    uint32_t keys_size = (*--keys_metas->end()).size + (*--keys_metas->end()).position;


    std::vector<const TupleRow *> result, all_results;
    std::vector<Partition> all_partitions;

    uint64_t *c_uuid = nullptr;
    char *buffer = nullptr;
    int32_t cluster_id = 0, offset = 0;
    int32_t *block = nullptr;
    int32_t half_int = 0;//-1 >> sizeof(int32_t)/2; //TODO be done properly

    SpaceFillingCurve::PartitionGenerator *partitions_it = this->partitioner.make_partitions_generator(metadata,
                                                                                                       nullptr);
    this->cache->flush_elements();

    while (!partitions_it->isDone()) {
        cluster_id = partitions_it->computeNextClusterId();
        buffer = (char *) malloc(keys_size);
        //UUID
        c_uuid = new uint64_t[2]{*storage_id, *(storage_id + 1)};
        //[0] time_and_version;
        //[1] clock_seq_and_node;
        memcpy(buffer, &c_uuid, sizeof(uint64_t *));
        offset = sizeof(uint64_t *);
        //Cluster id
        memcpy(buffer + offset, &cluster_id, sizeof(cluster_id));
        //We fetch the data
        const TupleRow *tr = new TupleRow(keys_metas, keys_size, buffer);
        result = read_cache->get_crow(tr);
        //build cluster
        all_results.insert(all_results.end(), result.begin(), result.end());
        for (const TupleRow *row:result) {
            block = (int32_t *) row->get_element(0);
            char **chunk = (char **) row->get_element(1);
            all_partitions.emplace_back(
                    Partition((uint32_t) cluster_id + half_int, (uint32_t) *block + half_int, *chunk));
        }
    }


    if (all_partitions.empty()) {
        throw ModuleException("no npy found on sys");
    }

    void *data = partitions_it->merge_partitions(metadata, all_partitions);

    for (const TupleRow *item:all_results) delete (item);

    delete (partitions_it);

    return data;
}
