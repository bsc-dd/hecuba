#include "SpaceFillingCurve.h"
#include "ModuleException.h"


/**
 * Base method does not make any partition
 * @param metas
 * @param data
 * @return
 */
SpaceFillingCurve::PartitionGenerator *
SpaceFillingCurve::make_partitions_generator(const ArrayMetadata *metas, void *data) {
    if (!metas) throw ModuleException("Array metadata not present");
    if (metas->partition_type == ZORDER_ALGORITHM) return new ZorderCurveGenerator(metas, data);
    return new SpaceFillingGenerator(metas, data);
}


SpaceFillingCurve::SpaceFillingGenerator::SpaceFillingGenerator() : done(true) {}

SpaceFillingCurve::SpaceFillingGenerator::SpaceFillingGenerator(const ArrayMetadata *metas, void *data) : done(false),
                                                                                                          metas(metas),
                                                                                                          data(data) {
    total_size = metas->elem_size;
    for (uint32_t dim:metas->dims) total_size *= dim;
}

Partition SpaceFillingCurve::SpaceFillingGenerator::getNextPartition() {
    if (data != nullptr && !done) {
        done = true;
        void *tobewritten = malloc(sizeof(uint64_t) + total_size);
        memcpy(tobewritten, &total_size, sizeof(uint64_t));
        memcpy((char *) tobewritten + sizeof(uint64_t), data, total_size);
        return {0, 0, tobewritten};
    }
    done = true;
    return {CLUSTER_END_FLAG, 0, nullptr};
}


int32_t SpaceFillingCurve::SpaceFillingGenerator::computeNextClusterId() {
    done = true;
    if (!done) return 0;
    return CLUSTER_END_FLAG;
}

void SpaceFillingCurve::SpaceFillingGenerator::simpleNextClusterId() {
    computeNextClusterId();
}

void *
SpaceFillingCurve::SpaceFillingGenerator::merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks, char* data) {
    uint64_t arrsize = metas->elem_size;

    for (uint32_t dim:metas->dims) arrsize *= dim;

    if (data== nullptr) {
        data = (char *) malloc(arrsize);
    }

    uint64_t block_size = arrsize; //metas->block_size;
    for (Partition part:chunks) {
        uint64_t *chunk_size = (uint64_t *) part.data;
        uint64_t offset = part.block_id * block_size;
        memcpy(data + offset, ((char *) part.data) + sizeof(uint64_t *), *chunk_size);
    }
    return data;
}


/*** Zorder (morton encoding) algorithms ***/

ZorderCurveGenerator::ZorderCurveGenerator() : done(true) {}

ZorderCurveGenerator::ZorderCurveGenerator(const ArrayMetadata *metas, void *data) : done(false), metas(metas),
                                                                                     data(data) {

    ndims = (uint32_t) metas->dims.size();
    //Compute the best fitting block
    //Make the block size multiple of the element size
    block_size = BLOCK_SIZE - (BLOCK_SIZE % metas->elem_size);
    //Compute the max number of elements per dimension as the ndims root of the block size
    row_elements = (uint32_t) std::floor(pow(block_size / metas->elem_size, (1.0 / ndims)));
    //TODO nth root returns an approximated value, which is later truncated by floor
    // Example: 125^(1.0/3) returns 4.9 -> 4: Correct is 5

    //Compute the final block size
    block_size = (uint64_t) pow(row_elements, ndims) * metas->elem_size;
    //Compute the number of blocks
    nblocks = 1;
    blocks_dim = std::vector<uint32_t>(ndims);
    for (uint32_t dim = 0; dim < ndims; ++dim) {
        blocks_dim[dim] = (uint32_t) std::ceil((double) metas->dims[dim] / row_elements);
        nblocks *= blocks_dim[dim];
    }

    block_dims = std::vector<uint32_t>(ndims, row_elements);
    bound_dims = std::vector<uint32_t>(ndims);
    //Create the blocks
    block_counter = 0;

}


uint64_t ZorderCurveGenerator::computeZorder(std::vector<uint32_t> cc) {
    uint64_t ndims = cc.size();
    uint64_t answer = 0;
    //(sizeof(uint64_t) * CHAR_BIT) equals to the number of bits in a uin64_t (8*8)
    uint32_t nbits = (sizeof(uint64_t) * CHAR_BIT) / ndims;
    for (uint64_t i = 0; i < nbits; ++i) {
        for (uint64_t dim_i = 0; dim_i < ndims; ++dim_i) {
            if (cc[dim_i] & ((uint64_t) 1 << i)) answer |= 1 << (ndims * i + dim_i);
        }
    }
    return answer;
}

std::vector<uint32_t> ZorderCurveGenerator::zorderInverse(uint64_t id, uint64_t ndims) {
    std::vector<uint32_t> ccs = std::vector<uint32_t>(ndims, 0);
    int32_t step = -1;
    for (uint32_t i = 0; i < sizeof(uint64_t) * CHAR_BIT; ++i) {
        if (i % ndims == 0) ++step;
        if ((id >> i & 1) == 1) ccs[i % ndims] |= 1 << step;
    }
    return ccs;
}

std::vector<uint32_t> ZorderCurveGenerator::getIndexes(uint64_t id, const std::vector<uint32_t> &dims) {
    uint64_t total_size = 1;
    for (uint32_t dim: dims) {
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

uint64_t
ZorderCurveGenerator::getIdFromIndexes(const std::vector<uint32_t> &dims, const std::vector<uint32_t> &indexes) {
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
void
ZorderCurveGenerator::tessellate(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size,
                                 char *data,
                                 char *output_data, char *output_data_end) {

    uint32_t elements_last_dim = block_dims[block_dims.size() - 1];
    if (dims.size() == 1) {
        if (output_data + elements_last_dim * elem_size > output_data_end) {
            throw ModuleException("Out of memory access copying an array block");
        }
        memcpy(output_data, data, elements_last_dim * elem_size);
    } else {
        //There are block_dims[0] subsets of the current dimension
        uint32_t elements_current_dim = block_dims[0];
        block_dims.erase(block_dims.begin());
        dims.erase(dims.begin());
        //Block_dims_prod = number of elements inside the current dimension of the block
        //Dims_prod = number of elements inside the current dimension of the array
        uint64_t block_dims_prod = elem_size;
        uint64_t dims_prod = elem_size;
        for (uint32_t i = 0; i < dims.size(); ++i) {
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


Partition ZorderCurveGenerator::getNextPartition() {
    if (block_counter == nblocks) return {CLUSTER_END_FLAG, 0, nullptr};

    char *output_data, *output_data_end;

    //Compute position in memory and chunks of data to copy
    std::vector<uint32_t> block_ccs = getIndexes(block_counter, blocks_dim);
    uint64_t zorder_id = computeZorder(block_ccs);

    //Block parameters
    uint32_t cluster_id = (uint32_t) (zorder_id >> CLUSTER_SIZE);
    uint64_t mask = (uint64_t) -1 >> (sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE);
    uint32_t block_id = (uint32_t) (zorder_id & mask);

    //if any element of the block_ccs is equal to dim_split -> is a limit of the array -> recompute block size
    bool bound = false;
    for (uint32_t i = 0; i < ndims; ++i) {
        if (block_ccs[i] == blocks_dim[i] - 1) bound = true;
        block_ccs[i] *= row_elements;
    }

    //Number of elements to skip until the coordinates
    uint64_t offset = getIdFromIndexes(metas->dims, block_ccs);

    //Compute the real offset as: position inside the array * sizeof(element)
    char *input_start = ((char *) data) + offset * metas->elem_size;

    if (!bound) {
        //In this case the block has size of row_elements in every_dimension
        //Create block
        output_data = (char *) malloc(block_size + sizeof(uint64_t));

        //copy the number of bytes
        memcpy(output_data, &block_size, sizeof(uint64_t));
        output_data += sizeof(uint64_t);
        output_data_end = output_data + block_size;
        //Copy the data
        tessellate(metas->dims, block_dims, metas->elem_size, input_start, output_data, output_data_end);

    } else {
        //The block is a limit of the array, and its size needs to be recomputed and adjusted

        //compute block size
        uint64_t bound_size = metas->elem_size;
        for (uint32_t i = 0; i < ndims; ++i) {
            //compute elem per dimension to be copied
            if (block_ccs[i] / row_elements != (blocks_dim[i] - 1)) {
                //Dimension isn't a limit, copy row_elements
                bound_dims[i] = row_elements;
            } else {
                //Is a limit, copy the remaining elements
                bound_dims[i] = (metas->dims[i] - (blocks_dim[i] - 1) * row_elements);
            }
            bound_size *= bound_dims[i];
        }

        //Create block
        output_data = (char *) malloc(bound_size + sizeof(uint64_t)); //chunk_size
        //Create block pointing to the memory

        memcpy(output_data, &bound_size, sizeof(uint64_t)); //copy the number of bytes
        output_data += sizeof(uint64_t);
        output_data_end = output_data + bound_size;
        //Copy the data
        tessellate(metas->dims, bound_dims, metas->elem_size, input_start, output_data, output_data_end);
    }

    ++block_counter;
    if (block_counter == nblocks) done = true;
    return {cluster_id, block_id, output_data - sizeof(uint64_t)};
}

void ZorderCurveGenerator::simpleNextClusterId() {
    block_counter += CLUSTER_SIZE << 1;
}



int32_t ZorderCurveGenerator::computeNextClusterId() {


    if (done || block_counter == nblocks) {
        done = true;
        return CLUSTER_END_FLAG;
    }

    std::vector<uint32_t> block_ccs = getIndexes(block_counter, blocks_dim);
    uint64_t zorder_id = computeZorder(block_ccs);
    //++block_counter;

    // Every cluster is made of 2^CLUSTER_SIZE blocks, we can skip these blocks
    simpleNextClusterId();
    if (block_counter == nblocks) done = true;
    //Block parameters
    return (uint32_t) (zorder_id >> CLUSTER_SIZE);
}

/***
 * @param dims Dimensions of the future array
 * @param block_shape Dimensions of the block of data passed
 * @param elem_size Size of every single element in the array
 * @param output_array Pointer to the beginning of the position of the array where data should be written
 * @param input_block Pointer to the beginning of the the block
 * @param input_block_end End of the memory allocated for the block
 */
void ZorderCurveGenerator::copy_block_to_array(std::vector<uint32_t> dims, std::vector<uint32_t> block_shape,
                                               uint32_t elem_size,
                                               char *output_array, char *input_block, char *input_block_end) {

    uint32_t elements_last_dim = block_shape[block_shape.size() - 1];
    if (dims.size() == 1) {
        if (input_block + elements_last_dim * elem_size > input_block_end) {
            throw ModuleException("Out of memory access copying an block into an array");
        }
        memcpy(output_array, input_block, elements_last_dim * elem_size);
    } else {
        //There are block_dims[0] subsets of the current dimension
        uint32_t elements_current_dim = block_shape[0];
        block_shape.erase(block_shape.begin());
        dims.erase(dims.begin());
        //Block_dims_prod = number of elements inside the current dimension of the block
        //Dims_prod = number of elements inside the current dimension of the array
        uint64_t block_dims_prod = elem_size;
        uint64_t dims_prod = elem_size;
        for (uint32_t i = 0; i < dims.size(); ++i) {
            dims_prod *= dims[i];
            block_dims_prod *= block_shape[i];
        }
        //Output_offset = Elements written in each subset
        uint64_t output_offset = (block_dims_prod / block_shape[block_shape.size() - 1]) * elements_last_dim;

        for (uint32_t subset = 0; subset < elements_current_dim; ++subset) {
            //Each subset is spaced as the product of its dimensions
            copy_block_to_array(dims, block_shape, elem_size, output_array + subset * dims_prod,
                                input_block + subset * output_offset, input_block_end);
        }
    }
}


void *ZorderCurveGenerator::merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks, char* data) {
    uint32_t ndims = (uint32_t) metas->dims.size();

    //Compute the best fitting block
    //Make the block size multiple of the element size
    uint64_t block_size = BLOCK_SIZE - (BLOCK_SIZE % metas->elem_size);
    //Compute the max number of elements per dimension as the ndims root of the block size
    uint32_t row_elements = (uint32_t) std::floor(pow(block_size / metas->elem_size, (1.0 / ndims)));
    //TODO nth root returns an approximated value, which is later truncated by floor
    // Example: 125^(1.0/3) returns 4.9 -> 4: Correct is 5

    //Compute the final block size
    block_size = (uint64_t) pow(row_elements, ndims) * metas->elem_size;

    //Compute the number of blocks and the final size of the array
    //Save the highest number of blocks for a dimension to later compute the maximum ZorderId
    uint64_t total_size = metas->elem_size;
    std::vector<uint32_t> blocks_dim(ndims);
    for (uint32_t dim = 0; dim < ndims; ++dim) {
        total_size *= metas->dims[dim];
        blocks_dim[dim] = (uint32_t) std::ceil((double) metas->dims[dim] / row_elements);
    }

    if (data== nullptr) {
        data = (char *) malloc(total_size);
    }

    //Shape of the average block
    std::vector<uint32_t> block_shape(ndims, row_elements);

    //For each partition compute the future position inside the new array
    //Achieved using the cluster_id and block_id to recompute the ZorderId
    for (Partition chunk : chunks) {
        uint64_t zorder_id = chunk.cluster_id << CLUSTER_SIZE | chunk.block_id;
        //Compute position in memory
        std::vector<uint32_t> ccs = zorderInverse(zorder_id, ndims); //Block coordinates
        //if any element of the ccs is equal to dim_split -> is a limit of the array -> recompute chunk
        bool bound = false;
        for (uint32_t i = 0; i < ndims; ++i) {
            if (ccs[i] == blocks_dim[i] - 1) bound = true;
        }

        //Scale coordinates to element coordinates
        for (uint32_t i = 0; i < ndims; ++i) {
            ccs[i] *= row_elements;
        }

        //Number of elements to skip until the coordinates
        uint64_t offset = getIdFromIndexes(metas->dims, ccs);
        char *output_start = data + offset * metas->elem_size;
        char *input = (char *) chunk.data;
        uint64_t *retrieved_block_size = (uint64_t *) input;
        input += +sizeof(uint64_t);
        char *input_ends = input + *retrieved_block_size;


        if (!bound) {

            if (*retrieved_block_size != block_size)
                throw ModuleException("Sth went wrong deciding "
                                      "the size of blocks while merging them into an array");


            copy_block_to_array(metas->dims, block_shape, metas->elem_size, output_start, input, input_ends);


        } else {

            //The block is a limit of the array, and its size needs to be recomputed and adjusted
            std::vector<uint32_t> bound_dims(ndims);
            //compute block size
            uint64_t bound_size = metas->elem_size;
            for (uint32_t i = 0; i < ndims; ++i) {
                //compute elem per dimension to be copied
                if (ccs[i] / row_elements != (blocks_dim[i] - 1)) {
                    //Dimension isn't a limit, copy row_elements
                    bound_dims[i] = (int32_t) row_elements;
                } else {
                    //Is a limit, copy the remaining elements
                    bound_dims[i] = (uint32_t) (metas->dims[i] - (blocks_dim[i] - 1) * row_elements);
                }
                bound_size *= bound_dims[i];
            }

            //Copy the data
            copy_block_to_array(metas->dims, bound_dims, metas->elem_size, output_start, input, input_ends);
        }
    }
    return data;
}


