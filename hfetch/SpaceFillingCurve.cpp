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

/***
 * @param elem_size Size of every single element in the array
 * @param block_dims Dimensions of the block being created
 * @param dims Dimensions of the given array
 * @param data Pointer to the beginning of the array data
 * @param output_data Pointer to the beginning of the memory allocated for the block
 * @param output_data_end End of the memory allocated for the block
 */
void ZorderCurve::tessellate(std::vector<int32_t> dims, std::vector<int32_t> block_dims, uint32_t elem_size, char *data,
                             char *output_data, char *output_data_end) const {

    uint64_t elements_last_dim = (uint64_t) block_dims[block_dims.size() - 1];
    if (dims.size() == 1) {
        if (output_data + elements_last_dim * elem_size > output_data_end) {
            throw ModuleException("Out of memory access copying an array block");
        }
        memcpy(output_data, data, elements_last_dim * elem_size);
    } else {
        //There are block_dims[0] subsets of the current dimension
        uint64_t elements_current_dim = (uint64_t) block_dims[0];
        block_dims.erase(block_dims.begin());
        dims.erase(dims.begin());
        //Block_dims_prod = number of elements inside the current dimension of the block
        //Dims_prod = number of elements inside the current dimension of the array
        uint64_t block_dims_prod = elem_size;
        uint64_t dims_prod = elem_size;
        for (int32_t i = 0; i < dims.size(); ++i) {
            dims_prod *= dims[i];
            block_dims_prod *= block_dims[i];
        }
        //Output_offset = Elements written in each subset
        uint64_t output_offset = (block_dims_prod / block_dims[block_dims.size() - 1]) * elements_last_dim;

        for (uint32_t subset = 0; subset < elements_current_dim; ++subset) {
            //Each subset is spaced as the product of its dimensions
            tessellate(dims, block_dims, elem_size, data + subset * dims_prod, output_data + subset * output_offset,
                       output_data_end);
        }
    }
}

std::vector<Partition> ZorderCurve::make_partitions(const ArrayMetadata *metas, void *data) const {
    uint32_t ndims = (uint32_t) metas->dims.size();

    char *output_data, *output_data_end;

    //Compute the best fitting block
    //Make the block size multiple of the element size
    uint64_t block_size = BLOCK_SIZE - (BLOCK_SIZE % metas->elem_size);
    //Compute the max number of elements per dimension as the ndims root of the block size
    uint64_t row_elements = (uint64_t) std::floor(pow(block_size / metas->elem_size, (1.0 / ndims)));
    //TODO nth root returns an approximated value, which is later truncated by floor
    // Example: 125^(1.0/3) returns 4.9 -> 4: Correct is 5

    //Compute the final block size
    block_size = (uint64_t) pow(row_elements, ndims) * metas->elem_size;

    //Compute the number of blocks
    //Save the highest number of blocks for a dimension to later compute the maximum ZorderId
    uint64_t max_blocks_in_dim = 0;
    uint64_t nblocks = 1;
    std::vector<uint64_t> blocks_dim(ndims);
    for (int32_t dim = 0; dim < ndims; ++dim) {
        blocks_dim[dim] = (uint64_t) std::ceil((double) metas->dims[dim] / row_elements);
        nblocks *= blocks_dim[dim];
        if (blocks_dim[dim] > max_blocks_in_dim) max_blocks_in_dim = blocks_dim[dim];
    }

    std::vector<int32_t> block_dims(ndims, row_elements);

    //Create the blocks
    std::vector<Partition> partitions = std::vector<Partition>(nblocks, {0, 0, nullptr});

    //Fill them with the data
    //Upper limit tells the maximum possible Zorder id
    uint64_t upper_limit = (uint64_t) 1 << ((uint64_t) std::ceil(std::log2(max_blocks_in_dim))) * ndims;
    uint64_t block_counter = 0;
    //For each ZorderId decide if the block will have data (the id can correspond to a block outside of the array)
    for (uint64_t zorder_id = 0; zorder_id < upper_limit; ++zorder_id) {

        //Compute position in memory and chunks of data to copy
        std::vector<uint32_t> ccs = zorderInverse(zorder_id, ndims); //Block coordinates
        //if any element of the ccs is equal to dim_split -> is a limit of the array -> recompute chunk
        bool outside = false, bound = false;
        for (uint32_t i = 0; i < ndims; ++i) {
            if (ccs[i] >= blocks_dim[i]) outside = true;
            else if (ccs[i] == blocks_dim[i] - 1) bound = true;
        }

        if (!outside) {
            if (block_counter >= nblocks) {
                throw ModuleException("Overflow: access more blocks than created partitioning array");
            }

            //Block parameters
            partitions[block_counter].cluster_id = (uint32_t) (zorder_id >> CLUSTER_SIZE);
            int64_t mask = -1 << (sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE);
            mask = (uint64_t) mask >> sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE;
            partitions[block_counter].block_id = (uint32_t) (zorder_id & mask);

            for (uint32_t i = 0; i < ndims; ++i) {
                ccs[i] *= row_elements;
            }

            //Number of elements to skip until the coordinates
            uint64_t offset = getIdFromIndexes(metas->dims, ccs);

            //Compute the real offset as: position inside the array * sizeof(element)
            char *input_start = ((char *) data) + offset * metas->elem_size;

            if (!bound) {
                //In this case the block has size of row_elements in every_dimension
                //Create block
                output_data = (char *) malloc(block_size + sizeof(uint64_t));

                //Create block pointing to the memory
                partitions[block_counter].data = output_data;
                //copy the number of bytes
                memcpy(output_data, &block_size, sizeof(uint64_t));
                output_data += sizeof(uint64_t);
                output_data_end = output_data + block_size;

                //Copy the data
                tessellate(metas->dims, block_dims,metas->elem_size , input_start, output_data, output_data_end);
            } else {
                //The block is a limit of the array, and its size needs to be recomputed and adjusted
                std::vector<int32_t> bound_dims(ndims);
                //compute block size
                uint64_t bound_size = metas->elem_size;
                for (uint32_t i = 0; i < ndims; ++i) {
                    //compute elem per dimension to be copied
                    if (ccs[i] / row_elements != (blocks_dim[i] - 1)) {
                        //Dimension isn't a limit, copy row_elements
                        bound_dims[i] = (int32_t) row_elements;
                    } else {
                        //Is a limit, copy the remaining elements
                        bound_dims[i] = (int32_t) (metas->dims[i] - (blocks_dim[i] - 1) * row_elements);
                    }
                    bound_size *= bound_dims[i];
                }

                //Create block
                output_data = (char *) malloc(bound_size + sizeof(uint64_t)); //chunk_size
                //Create block pointing to the memory
                partitions[block_counter].data = output_data;
                memcpy(output_data, &bound_size, sizeof(uint64_t)); //copy the number of bytes
                output_data += sizeof(uint64_t);
                output_data_end = output_data + bound_size;

                //Copy the data
                tessellate(metas->dims,bound_dims, metas->elem_size,input_start, output_data, output_data_end);
            }
            ++block_counter;
        }
    }
    return partitions;
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


