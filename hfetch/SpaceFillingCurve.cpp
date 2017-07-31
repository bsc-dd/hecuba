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

uint64_t computeZorder(std::vector<uint32_t> cc) {
    uint64_t ndims = cc.size();
    uint64_t accumulator = 0;
    uint64_t answer = 0;
    //(sizeof(uint64_t) * CHAR_BIT) equals to the number of bits in a uin64_t (8*8)
    for (uint64_t i = 0; i < (sizeof(uint64_t) * CHAR_BIT) / ndims; ++i) {
        accumulator = 0;
        for (uint64_t dim_i = 0; dim_i < ndims; ++dim_i) {
            accumulator |= ((cc[dim_i] & ((uint64_t) 1 << i)) << (2 * i + dim_i));
        }
        answer |= accumulator;
    }
    return answer;
}

std::vector<Partition> ZorderCurve::make_partitions(const ArrayMetadata *metas, void *data) const {
    /**
     * here we should implement some sort of algorithm which returns
     * the chunks of data with their block_id and cluster_id
     */

    //only for C like Nps

    uint64_t total_size = metas->elem_size;
    for (int32_t dim: metas->dims) {
        total_size *= dim;
    }
    uint64_t nblocks = total_size / BLOCK_SIZE;
    uint64_t offset = total_size % BLOCK_SIZE;
    uint64_t chunk_size = (total_size - offset) / nblocks;
    std::vector<Partition> parts = std::vector<Partition>();
    uint32_t block_id, cluster_id;
    char *output_data, *input_data;
    for (uint64_t block_i = 0; block_i < nblocks; ++block_i) {
        //Block parameters
        uint64_t blockandcluster = block_i;//computeZorder;
        block_id = (uint32_t) (blockandcluster >> 32);
        cluster_id = (uint32_t) blockandcluster;
        //Block memory
        output_data = (char *) malloc(chunk_size);
        //Create block pointing to the memory
        parts.push_back(Partition(block_id, cluster_id, output_data));
        //Fill the block
        input_data = (char *) data + block_i * BLOCK_SIZE;
        for (int32_t dim: metas->dims) {
            uint64_t write_row_size = dim / nblocks * metas->elem_size; //elements in the row_block * size
            memcpy(output_data, input_data, write_row_size);
            output_data += write_row_size;
            input_data += dim * metas->elem_size; //next dimension data
        }
    }
    return parts;
}


void *ZorderCurve::merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const {
    uint64_t arrsize = metas->elem_size;

    for (uint32_t dim:metas->dims) arrsize *= dim;

    char *data = (char *) malloc(arrsize);
    uint64_t block_size = arrsize; //metas->block_size;
    for (Partition part:chunks) {
        uint64_t *chunk_size = (uint64_t *) part.data;
        uint64_t offset = part.block_id * block_size;
        memcpy(data + offset, ((char *) part.data) + sizeof(uint64_t *), *chunk_size);
    }
    return data;
}


