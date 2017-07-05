#include "ArrayPartitioner.h"

/**
 * Base method does not make any partition
 * @param metas
 * @param data
 * @return
 */
std::vector<Partition> ArrayPartitioner::make_partitions(ArrayMetadata metas, void* data) const {
    uint64_t arrsize=3*5*sizeof(double);
    void *tobewritten = malloc(sizeof(uint64_t)+arrsize);
    memcpy(tobewritten,&arrsize,sizeof(uint64_t));

    memcpy((char*)tobewritten+sizeof(uint64_t),data,arrsize);
    return {Partition(0,0,tobewritten)};
}


std::vector<Partition> ZorderPartitioner::make_partitions(ArrayMetadata metas, void* data) const {
    /**
     * here we should implement some sort of algorithm which returns
     * the chunks of data with their block_id and cluster_id
     */
    return std::vector<Partition> {Partition(0,0,data)};
}

