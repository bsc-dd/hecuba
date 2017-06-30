#include "ArrayPartitioner.h"
#include "ModuleException.h"

/**
 * Base method does not make any partition
 * @param metas
 * @param data
 * @return
 */
std::vector<Partition> ArrayPartitioner::make_partitions(ArrayMetadata metas, void* data) {
    return std::vector<Partition> {Partition(0,0,data)};
}


std::vector<Partition> ZorderPartitioner::make_partitions(ArrayMetadata metas, void* data) {
    /**
     * here we should implement some sort of algorithm which returns
     * the chunks of data with their block_id and cluster_id
     */
    return std::vector<Partition> {Partition(0,0,data)};
}

