from hecuba import StorageDict
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on


class DataDict(StorageDict):
    '''
    @TypeSpec dict <<a:int>,b:int>
    '''


@task()
def compute_stats(block: DataDict):
    nitems = 0
    total_k = 0
    total_v = 0
    for k in block.keys():
        total_k = total_k + k
    for v in block.values():
        total_v = total_v + v
    for k,v in block.items():
        nitems = nitems + 1
    return nitems, total_k, total_v


if __name__ == '__main__':
    # Populate
    dataset = DataDict('test.zero')
    for i in range(10**3):
        dataset[i]=i*5

    results = []
    for block in dataset.split():
        block_stats = compute_stats(block)
        results.append(block_stats)

    # Option A: Bring everything to master
    results = [compss_wait_on(block_stat) for block_stat in results]

    # Option B: Create new task to merge

    # After asserting results, assert number of splits
    # TODO Flush on split
