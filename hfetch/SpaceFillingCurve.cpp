#include "SpaceFillingCurve.h"

/**
 * Base method does not make any partition
 * @param metas
 * @param data
 * @return
 */
std::vector<Partition> SpaceFillingCurve::make_partitions(const ArrayMetadata *metas, void *data) const {
    uint64_t arrsize = metas->elem_size;
    for (int32_t dim:metas->dims) arrsize *= dim;
    void *tobewritten = malloc(sizeof(uint64_t) + arrsize);
    memcpy(tobewritten, &arrsize, sizeof(uint64_t));

    memcpy((char *) tobewritten + sizeof(uint64_t), data, arrsize);
    return {Partition(0, 0, tobewritten)};
}


void *SpaceFillingCurve::merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const {
    uint64_t arrsize = metas->elem_size;

    for (int32_t dim:metas->dims) arrsize *= dim;

    char *data = (char *) malloc(arrsize);
    uint64_t block_size = arrsize; //metas->block_size;
    for (Partition part:chunks) {
        uint64_t *chunk_size = (uint64_t *) part.data;
        uint64_t offset = part.block_id * block_size;
        memcpy(data + offset, ((char *) part.data) + sizeof(uint64_t *), *chunk_size);
    }
    return data;
}


std::vector<Partition> ZorderCurve::make_partitions(const ArrayMetadata *metas, void *data) const {
    /**
     * here we should implement some sort of algorithm which returns
     * the chunks of data with their block_id and cluster_id
     */
    uint64_t arrsize = metas->elem_size;
    for (int32_t dim:metas->dims) arrsize *= dim;
    void *tobewritten = malloc(sizeof(uint64_t) + arrsize);
    memcpy(tobewritten, &arrsize, sizeof(uint64_t));

    memcpy((char *) tobewritten + sizeof(uint64_t), data, arrsize);
    return {Partition(0, 0, tobewritten)};
}


void *ZorderCurve::merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const {
    uint64_t arrsize = metas->elem_size;

    for (int32_t dim:metas->dims) arrsize *= dim;

    char *data = (char *) malloc(arrsize);
    uint64_t block_size = arrsize; //metas->block_size;
    for (Partition part:chunks) {
        uint64_t *chunk_size = (uint64_t *) part.data;
        uint64_t offset = part.block_id * block_size;
        memcpy(data + offset, ((char *) part.data) + sizeof(uint64_t *), *chunk_size);
    }
    return data;
}


