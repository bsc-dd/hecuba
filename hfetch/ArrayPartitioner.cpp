#include "ArrayPartitioner.h"
#include "ModuleException.h"

/**
 * Base method does not make any partition
 * @param metas
 * @param data
 * @return
 */
std::vector<Partition> ArrayPartitioner::make_partitions(const ArrayMetadata *metas, void* data) const {
    uint64_t arrsize=1;

    for (int32_t dim:metas->dims)arrsize*=dim;
    arrsize*=sizeof(int32_t); //TODO pass from metas->type to metas->size

    void *tobewritten = malloc(sizeof(uint64_t)+arrsize);
    memcpy(tobewritten,&arrsize,sizeof(uint64_t));

    memcpy((char*)tobewritten+sizeof(uint64_t),data,arrsize);
    return {Partition(0,0,tobewritten)};
}


std::vector<Partition> ZorderPartitioner::make_partitions(const ArrayMetadata *metas, void* data) const {
    /**
     * here we should implement some sort of algorithm which returns
     * the chunks of data with their block_id and cluster_id
     */
    return std::vector<Partition> {Partition(0,0,data)};
}

void* ArrayPartitioner::merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const {
    uint64_t arrsize=1;

    for (int32_t dim:metas->dims)arrsize*=dim;
    arrsize*=sizeof(int32_t); //TODO pass from metas->type to metas->size
    char *data = (char*) malloc(arrsize);
    for (Partition part:chunks) {
        uint64_t *chunk_size = (uint64_t *) part.data;
        uint64_t offset = part.block_id*(*chunk_size);
        memcpy(data+offset,((char*)part.data)+sizeof(uint64_t*),metas->block_size);
    }
    return data;
}

void* ZorderPartitioner::merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const {
    throw ModuleException("Not implemented merge partition zorder");
}


