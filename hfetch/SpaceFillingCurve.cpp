#include "SpaceFillingCurve.h"
#include "ModuleException.h"

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

std::vector<uint32_t> ZorderCurve::zorderInverse(uint64_t id, uint64_t ndims) const {
    std::vector<uint32_t> ccs = std::vector<uint32_t>(ndims, 0);
    int32_t step = -1;
    for (uint32_t i = 0; i < sizeof(uint64_t) * CHAR_BIT; ++i) {
        if (i % ndims == 0) ++step;
        if ((id >> i & 1) == 1) ccs[i % ndims] |= 1 << step;
    }
    return ccs;
}

std::vector<uint32_t> ZorderCurve::getIndexes(uint64_t id, const std::vector<int32_t> &dims) const {
    uint64_t total_size = 1;
    for (int32_t dim: dims) {
        total_size *= dim;
    }
    total_size /= dims[0];

    std::vector<uint32_t> ccs = std::vector<uint32_t>(dims.size());
    uint32_t i = 0;
    for (; i < ccs.size() - 1; ++i) {
        std::ldiv_t dv = std::div((int64_t) id, total_size);
        ccs[i] = (uint32_t) dv.quot;
        total_size /= dims[i + 1];
        id = (uint64_t) dv.rem;
    }
    ccs[i] = (uint32_t) id;
    return ccs;
}

uint64_t ZorderCurve::getIdFromIndexes(const std::vector<int32_t> &dims, const std::vector<uint32_t> &indexes) const {
    uint64_t id = *(--indexes.end());
    uint64_t accumulator = 1;
    for (uint64_t i = dims.size() - 1; i > 0; --i) {
        accumulator *= dims[i];
        id += accumulator * indexes[i - 1];
    }
    return id;
}


std::vector<Partition> ZorderCurve::make_partitions(const ArrayMetadata *metas, void *data) const {
    uint32_t ndims = (uint32_t) metas->dims.size();

    uint64_t dims_product = 1;
    for (uint64_t i = 1; i < metas->dims.size(); ++i) {
        dims_product *= metas->dims[i];
    }

    uint64_t total_size = metas->elem_size;
    for (int32_t dim: metas->dims) {
        total_size *= dim;
    }
    char *input_end = ((char *) data) + total_size;
    char *output_data, *output_data_end;

    // Compute the best fitting block
    uint64_t b = BLOCK_SIZE;
    //if (total_size < b) b = total_size;
    uint64_t block_size = b - (b % metas->elem_size);
    uint64_t row_elements = (uint64_t) std::floor(pow(block_size / metas->elem_size, (1.0 / metas->dims.size())));
    block_size = (uint64_t) pow(row_elements, ndims) * metas->elem_size;

    //Compute the number of blocks
    uint64_t max_block_dim = 0;
    uint64_t nblocks = 1;
    std::vector <uint64_t> blocks_dim(ndims);
    for (int32_t dim = 0; dim<ndims; ++dim) {
        blocks_dim[dim] = (uint64_t) std::ceil((double)metas->dims[dim]/row_elements);
        nblocks *= blocks_dim[dim];
        if (blocks_dim[dim]>max_block_dim) max_block_dim = blocks_dim[dim];
    }

    //Create the blocks
    std::vector<Partition> parts = std::vector<Partition>(nblocks, {0, 0, nullptr});

    //Compute offsets to copy data
    uint64_t row_elements_size = row_elements * metas->elem_size;
    uint64_t row_offset = dims_product * metas->elem_size;

    //Fill them with the data
    uint64_t upper_limit = (uint64_t) 1 << ((uint64_t) std::ceil(std::log2(max_block_dim))) * ndims;
    uint64_t block_counter = 0;

    for (uint64_t zorder_id = 0; zorder_id < upper_limit; ++zorder_id) {

        //Compute position in memory and chunks of data to copy
        std::vector<uint32_t> ccs = zorderInverse(zorder_id, metas->dims.size()); //Block coordinates
        std::vector<uint32_t> original_ccs(ccs);
        //if any element of the ccs is equal to dim_split -> is a limit of the array -> recompute chunk
        bool outside = false, bound = false;
        for (uint32_t i = 0; i<ndims; ++i) {
            if (ccs[i] >= blocks_dim[i]) outside = true;
            else if (ccs[i] == blocks_dim[i] - 1) bound = true;
        }

        if (!outside) {
            if (block_counter >= nblocks) {
                throw ModuleException("Overflow: access more blocks than created partitioning array");
            }

            //Block parameters
            parts[block_counter].cluster_id = (uint32_t) (zorder_id >> CLUSTER_SIZE);
            int64_t mask = -1 << (sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE);
            mask = (uint64_t) mask >> sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE;
            parts[block_counter].block_id = (uint32_t) (zorder_id & mask);

            //Transform coordinates; scale them from block_coordinates
            //to element position
            for (uint32_t i = 0; i < ccs.size(); ++i) {
                ccs[i] *= row_elements;
            }

            uint64_t offset = getIdFromIndexes(metas->dims, ccs);

            //Compute the real offset as: position inside the array * sizeof(singleelement)
            char *input_start = ((char *) data) + offset * metas->elem_size;

            if (!bound) {
                if (input_start + row_elements_size > input_end)
                    throw ModuleException("Overflow reading np");

                //Create block
                output_data = (char *) malloc(block_size + sizeof(uint64_t));

                //Create block pointing to the memory
                parts[block_counter].data = output_data;
                //copy the number of bytes
                memcpy(output_data, &block_size, sizeof(uint64_t));
                output_data += sizeof(uint64_t);
                output_data_end = output_data + block_size;

                //TODO when dimensions>3 require a different way to iterate over data
                while (output_data < output_data_end) {
                    memcpy(output_data, input_start, row_elements_size);
                    input_start += row_offset;
                    output_data += row_elements_size;
                }
            } else {
                //bound

                //compute block size
                uint64_t bound_size = metas->elem_size;
                for (uint32_t i = 0; i < metas->dims.size(); ++i) {
                    //compute elem per dimension to be copied
                    if (original_ccs[i] != (blocks_dim[i] - 1)) {
                        bound_size *= row_elements;
                    } else {
                        bound_size *= (metas->dims[i] - (blocks_dim[i] - 1) * row_elements);
                    }
                }

                uint64_t single_copy_size;
                if (original_ccs[0] != (blocks_dim[0] - 1)) single_copy_size=bound_size/row_elements;
                else single_copy_size = bound_size/(metas->dims[0] - (blocks_dim[0] - 1) * row_elements);

                //Create block
                output_data = (char *) malloc(bound_size + sizeof(uint64_t)); //chunk_size
                //Create block pointing to the memory
                parts[block_counter].data = output_data;
                memcpy(output_data, &bound_size, sizeof(uint64_t)); //copy the number of bytes
                output_data += sizeof(uint64_t);
                output_data_end = output_data + bound_size;

                while (output_data < output_data_end) {
                    //last_dim*metas->elem_size-(block_size/product_dims*metas->elem_size);
                    memcpy(output_data, input_start, single_copy_size);
                    input_start += row_offset;
                    output_data += single_copy_size;
                }
            }
            ++block_counter;
        }
    }
    
    return parts;
}


void *ZorderCurve::merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const {
    uint64_t arrsize = metas->elem_size;

    for (uint32_t dim:metas->dims) arrsize *= dim;

    char *data = (char *) malloc(arrsize);
    //TODO merge partitions
    /*uint64_t block_size = arrsize; //metas->block_size;
    for (Partition part:chunks) {
        uint64_t *chunk_size = (uint64_t *) part.data;
        uint64_t offset = part.block_id * block_size;
        memcpy(data + offset, ((char *) part.data) + sizeof(uint64_t *), *chunk_size);
    }*/
    return data;
}


