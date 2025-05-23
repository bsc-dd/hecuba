#include "SpaceFillingCurve.h"
#include "ModuleException.h"
#include <math.h>


/**
 * Base method does not make any partition
 * @param metas
 * @param data
 * @return
 */
SpaceFillingCurve::PartitionGenerator *
SpaceFillingCurve::make_partitions_generator(const ArrayMetadata &metas, void *data) {
    if (metas.partition_type == ZORDER_ALGORITHM) return new ZorderCurveGenerator(metas, data);
    if (metas.partition_type == FORTRANORDER) return new FortranOrderGenerator(metas, data);
    return new SpaceFillingGenerator(metas, data);
}

SpaceFillingCurve::PartitionGenerator *
SpaceFillingCurve::make_partitions_generator(const ArrayMetadata &metas, void *data,
                                             std::list<std::vector<uint32_t> > &coord) {
    if (metas.partition_type == ZORDER_ALGORITHM) return new ZorderCurveGeneratorFiltered(metas, data, coord);
    if (metas.partition_type == FORTRANORDER) return new FortranOrderGeneratorFiltered(metas, data, coord);
    return new SpaceFillingGenerator(metas, data);
}

SpaceFillingCurve::SpaceFillingGenerator::SpaceFillingGenerator() : done(true) {}

SpaceFillingCurve::SpaceFillingGenerator::SpaceFillingGenerator(const ArrayMetadata &metas, void *data) : done(false),
                                                                                                          metas(metas),
                                                                                                          data(data) {
    total_size = metas.elem_size;
    for (uint32_t dim:metas.dims) total_size *= dim;
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
PartitionIdxs SpaceFillingCurve::SpaceFillingGenerator::getNextPartitionIdxs() {
    return {CLUSTER_END_FLAG, CLUSTER_END_FLAG, 0, {}}; // NOT IMPLEMENTED
}


int32_t SpaceFillingCurve::SpaceFillingGenerator::computeNextClusterId() {
    done = true;
    if (!done) return 0;
    return CLUSTER_END_FLAG;
}

void
SpaceFillingCurve::SpaceFillingGenerator::merge_partitions(const ArrayMetadata &metas, std::vector<Partition> chunks,
                                                           void *data) {
    uint64_t arrsize = metas.elem_size;

    for (uint32_t dim:metas.dims) arrsize *= dim;

    uint64_t block_size = arrsize; //metas->block_size;
    for (Partition part:chunks) {
        uint64_t *chunk_size = (uint64_t *) part.data;
        uint64_t offset = part.block_id * block_size;
        memcpy(static_cast<char *>(data) + offset, ((char *) part.data) + sizeof(uint64_t *), *chunk_size);
    }
}

uint32_t SpaceFillingCurve::SpaceFillingGenerator::getBlockID(std::vector<uint32_t> cc) {
    return 0;
}
uint32_t SpaceFillingCurve::SpaceFillingGenerator::getClusterID(std::vector<uint32_t> cc) {
    return 0;
}

/*** Zorder (morton encoding) algorithms ***/

/*
 *   block_counter >--- getIndexes() --> block_coords >-- computeZorder() --> zorderId
 *             ^                            v    ^                              v
 *             |                            |    |                              |
 *              \______ getBlockCounter()__/      \______ zorderInverse()______/
 */
ZorderCurveGenerator::ZorderCurveGenerator() : done(true) {}

ZorderCurveGenerator::ZorderCurveGenerator(const ArrayMetadata &metas, void *data) : done(false), metas(metas),
                                                                                     data(data) {
    ndims = (uint32_t) metas.dims.size();

    nreddims = 2;

    if (ndims<2)nreddims=ndims;
    //Compute the best fitting block
    //Make the block size multiple of the element size
    block_size = BLOCK_SIZE - (BLOCK_SIZE % metas.elem_size);
    //Compute the max number of elements per dimension as the ndims root of the block size
    row_elements = (uint32_t) std::floor(pow(block_size / metas.elem_size, (1.0 / ndims)));
    //TODO nth root returns an approximated value, which is later truncated by floor
    // Example: 125^(1.0/3) returns 4.9 -> 4: Correct is 5

    //Compute the final block size
    block_size = (uint64_t) pow(row_elements, ndims) * metas.elem_size;

    //Compute the number of blocks
    nblocks = 1;
    nclusters = 1;
    blocks_dim = std::vector<uint32_t>(ndims);
    clusters_dim = std::vector<uint32_t>(ndims);
    for (uint32_t dim = 0; dim < ndims; ++dim) {
        blocks_dim[dim] = (uint32_t) std::ceil((double) metas.dims[dim] / row_elements);
        if (dim<2){
            clusters_dim[dim] = (uint32_t) (blocks_dim[dim]+1)/2;
        } else{
            clusters_dim[dim] = (uint32_t) (blocks_dim[dim]);
        }
        nblocks *= blocks_dim[dim];
        nclusters *=clusters_dim[dim];
    }

    block_dims = std::vector<uint32_t>(ndims, row_elements);
    bound_dims = std::vector<uint32_t>(ndims);
    //Create the blocks
    block_counter = 0;
    cluster_counter = 0;

}


uint64_t ZorderCurveGenerator::computeZorder(std::vector<uint32_t> cc) {
    uint64_t ndims = cc.size();
    uint64_t answer = 0;
    //(sizeof(uint64_t) * CHAR_BIT) equals to the number of bits in a uin64_t (8*8)
    //std::cout<< "ZorderCurveGenerator::computeZorder cc={";
    //for (uint64_t i = 0; i < ndims; ++i) {
    //    std::cout<< cc[i] << ", ";
    //}
    uint32_t nbits = (sizeof(uint64_t) * CHAR_BIT) / ndims;
    for (uint64_t i = 0; i < nbits; ++i) {
        for (uint64_t dim_i = 0; dim_i < ndims; ++dim_i) {
            if (cc[dim_i] & ((uint64_t) 1 << i)) answer |= (((uint64_t)1) << (ndims * i + dim_i));
        }
    }
    //std::cout<< "} => " << answer << std::endl;
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

uint64_t ZorderCurveGenerator::getBlockCounter(std::vector<uint32_t> ccs, const std::vector<uint32_t> &dims) {
    uint64_t total_size = 1;
    uint64_t valor=0;

    for (int64_t i = ccs.size()-1; i>=0; i--){
        valor = ccs[i]*total_size+valor;
        total_size *= dims[i];
    }
    return valor;
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
    //std::cout<< " ZorderCurveGenerator::getIdFromIndexes dims =";
    //for (uint64_t i = dims.size() - 1; i >= 0; --i) {
    //    std::cout<< dims[i] << ", ";
    //}
    //std::cout<<" idxs = ";
    //for (uint64_t i = indexes.size() - 1; i >= 0; --i) {
    //    std::cout<< indexes[i] << ", ";
    //}
    //std::cout<< std::endl;

    uint64_t id = *(--indexes.end());
    uint64_t accumulator = 1;
    for (uint64_t i = dims.size() - 1; i > 0; --i) {
        accumulator *= dims[i];
        id += accumulator * indexes[i - 1];
    }
    //std::cout<< " ZorderCurveGenerator::getIdFromIndexes offset ="<<id<<std::endl;
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

/** 
 *  Args:
 *      block_ccs   Vector of indexes to blocks
 *  Returns:
 *      cluster_id  (CLUSTER_END_FLAG if done)
 *      block_id
 *      block_size
 *  PRE: Always called with block_counter < nblocks
 */
 /*
std::tuple<int32_t,int32_t> ZorderCurveGenerator::getNextBlockIds(std::vector<uint32_t> * block_ccs){

    uint64_t zorder_id;

    if (block_ccs == nullptr){
        std::vector<uint32_t> ix_blocks = getIndexes(block_counter, blocks_dim);
        zorder_id = computeZorder(ix_blocks);
    }
    else {
        zorder_id = computeZorder(*block_ccs);
    }

    //Block parameters
    uint32_t cluster_id = (uint32_t) (zorder_id >> CLUSTER_SIZE);
    uint64_t mask = (uint64_t) -1 >> (sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE);
    uint32_t block_id = (uint32_t) (zorder_id & mask);
    std::cout<< " JCOSTA cluster_id "<<cluster_id<< " block_id " << block_id << std::endl;
    return std::make_tuple(cluster_id, block_id);
}
*/

uint32_t ZorderCurveGenerator::getBlockID(std::vector<uint32_t> cc) {
    uint64_t zorder_id = computeZorder(cc);
    uint64_t mask = (uint64_t) -1 >> (sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE);
    uint32_t block_id = (uint32_t) (zorder_id & mask);
    return block_id;
}

uint32_t ZorderCurveGenerator::getClusterID(std::vector<uint32_t> cc) {
    uint64_t zorder_id = computeZorder(cc);
    uint32_t cluster_id = (uint32_t) (zorder_id >> CLUSTER_SIZE);
    return cluster_id;
}

Partition ZorderCurveGenerator::getNextPartition() {

    char *output_data, *output_data_end;

    //std::cout<< "ZorderCurveGenerator::getNextPartition block_counter " << block_counter << std::endl;

    //std::cout<< "ZorderCurveGenerator::getNextPartition blocks_dim={ ";
    //for(uint64_t i=0; i< blocks_dim.size(); i++) {
    //    std::cout<<blocks_dim[i]<<", ";
    //}
    //std::cout<<"}"<<std::endl;

    //Compute position in memory and chunks of data to copy
    std::vector<uint32_t> block_ccs = getIndexes(block_counter, blocks_dim);
    uint64_t zorder_id = computeZorder(block_ccs);

    //Block parameters
    uint32_t cluster_id = (uint32_t) (zorder_id >> CLUSTER_SIZE);
    uint64_t mask = (uint64_t) -1 >> (sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE);
    uint32_t block_id = (uint32_t) (zorder_id & mask);

    ++block_counter;
    //if (block_counter == nblocks) done = true;

    if (data == NULL)
        return {cluster_id, block_id, nullptr};


    //if any element of the block_ccs is equal to dim_split -> is a limit of the array -> recompute block size
    bool bound = false;
    for (uint32_t i = 0; i < ndims; ++i) {
        if (block_ccs[i] == blocks_dim[i] - 1) bound = true;
        block_ccs[i] *= row_elements;
    }

    //Number of elements to skip until the coordinates
    uint64_t offset = getIdFromIndexes(metas.dims, block_ccs);

    //Compute the real offset as: position inside the array * sizeof(element)
    char *input_start = ((char *) data) + offset * metas.elem_size;

    if (!bound) {
        //In this case the block has size of row_elements in every_dimension
        //Create block
        output_data = (char *) malloc(block_size + sizeof(uint64_t));

        //copy the number of bytes
        memcpy(output_data, &block_size, sizeof(uint64_t));
        output_data += sizeof(uint64_t);
        output_data_end = output_data + block_size;
        //Copy the data
        tessellate(metas.dims, block_dims, metas.elem_size, input_start, output_data, output_data_end);

    } else {
        //The block is a limit of the array, and its size needs to be recomputed and adjusted

        //compute block size
        uint64_t bound_size = metas.elem_size;
        for (uint32_t i = 0; i < ndims; ++i) {
            //compute elem per dimension to be copied
            if (block_ccs[i] / row_elements != (blocks_dim[i] - 1)) {
                //Dimension isn't a limit, copy row_elements
                bound_dims[i] = row_elements;
            } else {
                //Is a limit, copy the remaining elements
                bound_dims[i] = (metas.dims[i] - (blocks_dim[i] - 1) * row_elements);
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
        tessellate(metas.dims, bound_dims, metas.elem_size, input_start, output_data, output_data_end);
    }

    return {cluster_id, block_id, output_data - sizeof(uint64_t)};
}

PartitionIdxs ZorderCurveGenerator::getNextPartitionIdxs() {

    std::vector<uint32_t> ix_blocks = getIndexes(block_counter, blocks_dim);
    uint64_t zorder_id = computeZorder(ix_blocks);
    uint32_t cluster_id = (uint32_t) (zorder_id >> CLUSTER_SIZE);
    uint64_t mask = (uint64_t) -1 >> (sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE);
    uint32_t block_id = (uint32_t) (zorder_id & mask);

    block_counter ++;
    if (block_counter == nblocks) done = true;
    return {zorder_id, cluster_id, block_id, ix_blocks};
}



int32_t ZorderCurveGenerator::computeNextClusterId() {
    if (done || block_counter == nclusters) {
        done = true;
        return CLUSTER_END_FLAG;
    }

    std::vector<uint32_t> block_ccs = getIndexes(block_counter, clusters_dim);
    for (uint32_t i =0; i<nreddims; i++){
        block_ccs[i]*=2;
    }
    uint64_t zorder_id = computeZorder(block_ccs);

    block_counter++;

    if (block_counter == nclusters) done = true;
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


void ZorderCurveGenerator::merge_partitions(const ArrayMetadata &metas, std::vector<Partition> chunks, void *data) {
    uint32_t ndims = (uint32_t) metas.dims.size();

    //Compute the best fitting block
    //Make the block size multiple of the element size
    uint64_t block_size = BLOCK_SIZE - (BLOCK_SIZE % metas.elem_size);
    //Compute the max number of elements per dimension as the ndims root of the block size
    uint32_t row_elements = (uint32_t) std::floor(pow(block_size / metas.elem_size, (1.0 / ndims)));
    //TODO nth root returns an approximated value, which is later truncated by floor
    // Example: 125^(1.0/3) returns 4.9 -> 4: Correct is 5

    //Compute the final block size
    block_size = (uint64_t) pow(row_elements, ndims) * metas.elem_size;

    //Compute the number of blocks and the final size of the array
    //Save the highest number of blocks for a dimension to later compute the maximum ZorderId
    uint64_t total_size = metas.elem_size;
    std::vector<uint32_t> blocks_dim(ndims);
    for (uint32_t dim = 0; dim < ndims; ++dim) {
        total_size *= metas.dims[dim];
        blocks_dim[dim] = (uint32_t) std::ceil((double) metas.dims[dim] / row_elements);
    }

    //Shape of the average block
    std::vector<uint32_t> block_shape(ndims, row_elements);

    //For each partition compute the future position inside the new array
    //Achieved using the cluster_id and block_id to recompute the ZorderId
    for (Partition chunk : chunks) {
	uint64_t zorder_id = (uint64_t)chunk.cluster_id << CLUSTER_SIZE | (uint64_t)chunk.block_id;
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
        uint64_t offset = getIdFromIndexes(metas.dims, ccs);
        char *output_start = static_cast<char *>(data) + offset * metas.elem_size;
        char *input = (char *) chunk.data;
        uint64_t *retrieved_block_size = (uint64_t *) input;
        input += +sizeof(uint64_t);
        char *input_ends = input + *retrieved_block_size;


        if (!bound) {

            if (*retrieved_block_size != block_size)
                throw ModuleException("Sth went wrong deciding "
                                      "the size of blocks while merging them into an array");


            copy_block_to_array(metas.dims, block_shape, metas.elem_size, output_start, input, input_ends);


        } else {

            //The block is a limit of the array, and its size needs to be recomputed and adjusted
            std::vector<uint32_t> bound_dims(ndims);
            //compute block size
            uint64_t bound_size = metas.elem_size;
            for (uint32_t i = 0; i < ndims; ++i) {
                //compute elem per dimension to be copied
                if (ccs[i] / row_elements != (blocks_dim[i] - 1)) {
                    //Dimension isn't a limit, copy row_elements
                    bound_dims[i] = (int32_t) row_elements;
                } else {
                    //Is a limit, copy the remaining elements
                    bound_dims[i] = (uint32_t) (metas.dims[i] - (blocks_dim[i] - 1) * row_elements);
                }
                bound_size *= bound_dims[i];
            }

            //Copy the data
            copy_block_to_array(metas.dims, bound_dims, metas.elem_size, output_start, input, input_ends);
        }
    }
}

int32_t ZorderCurveGeneratorFiltered::computeNextClusterId() {
    uint32_t zorder = (uint32_t) (computeZorder(coord.front()) >> CLUSTER_SIZE);
    coord.erase(coord.begin());
    return zorder;
}

Partition ZorderCurveGeneratorFiltered::getNextPartition() {
    block_counter = getBlockCounter(coord.front(), blocks_dim);
    coord.erase(coord.begin());
    return ZorderCurveGenerator::getNextPartition();
}

bool ZorderCurveGeneratorFiltered::isDone() {
    if (coord.empty()) done = true;
    return done;

}

ZorderCurveGeneratorFiltered::ZorderCurveGeneratorFiltered(const ArrayMetadata &metas, void *data,
                                                           std::list<std::vector<uint32_t> > &coord)
        : ZorderCurveGenerator(metas, data) {
    this->coord = coord;
}


/************************************************************/
/* FORTRAN ORDER */
/*****************/



FortranOrderGenerator::FortranOrderGenerator() : done(true) {}

FortranOrderGenerator::FortranOrderGenerator(const ArrayMetadata &metas, void *data) : done(false), metas(metas),
                                                                                     data(data) {
    ndims = (uint32_t) metas.dims.size();

    nreddims = 2;

    if (ndims<2)nreddims=ndims;

    //Compute the best fitting block
    //Make the block size multiple of the element size
    block_size = BLOCK_SIZE - (BLOCK_SIZE % metas.elem_size);
    //Compute the max number of elements per dimension as the ndims root of the block size
    row_elements = (uint32_t) std::floor(pow(block_size / metas.elem_size, (1.0 / ndims)));
    //TODO nth root returns an approximated value, which is later truncated by floor
    // Example: 125^(1.0/3) returns 4.9 -> 4: Correct is 5
    //std::cout<< "row_elements for "<<ndims<<" dimensions -> "<<row_elements<<std::endl;

    //Compute the final block size
    block_size = (uint64_t) pow(row_elements, ndims) * metas.elem_size;

    //Compute the number of blocks
    nblocks = 1;
    blocks_dim = std::vector<uint32_t>(ndims);
    for (uint32_t dim = 0; dim < ndims; ++dim) {
        blocks_dim[dim] = (uint32_t) std::ceil((double) metas.dims[dim] / row_elements);
        nblocks   *= blocks_dim[dim];
        //std::cout<< "blocks_dim["<<dim<<"] = "<<blocks_dim[dim]<<std::endl;
    }
    nclusters = blocks_dim[blocks_dim.size()-1]; //Number of blocks in the last dimension

    block_dims = std::vector<uint32_t>(ndims, row_elements);
    bound_dims = std::vector<uint32_t>(ndims);
    //Create the blocks
    block_counter   = 0;
    cluster_counter = 0;

}


/***
 * Given a block coordinates calculate its linear number:
 * Example:
 *      (+)==========+           () coordenada inicial del numpy
 *       I 0 . 1 . 2 I <-- zorder
 *       I0,0.0,1.0,2I <-- block coordinates
 *       I---+---+---I
 *       I 3 . 4 . 5 I
 *       I1,0.1,1.1,2I
 *       I---+---+---I
 *       I 6 . 7 . 8 I
 *       I2,0.2,1.2,2I
 *       +===========+
 * @param cc: Vector of coordinates for the block
 * @return The Zorder for 'cc'
 */
uint64_t FortranOrderGenerator::computeZorder(std::vector<uint32_t> cc) {
    uint64_t ndims = cc.size();
    uint64_t answer = 0;
    //std::cout<< "FortranOrderGenerator::computeZorder cc={";
    //for (uint64_t i = 0; i < ndims; ++i) {
    //    std::cout<< cc[i] << ", ";
    //}
    for (uint64_t i = 0; i < ndims-1; ++i) {
        answer += cc[i] * blocks_dim[i+1];
    }
    answer += cc[ndims-1];
    //std::cout<< "} => " << answer << std::endl;
    return answer;
}

/* Given a block_counter return its initial coordinates */
std::vector<uint32_t> FortranOrderGenerator::zorderInverse(uint64_t id, uint64_t ndims) {
    std::vector<uint32_t> ccs = std::vector<uint32_t>(ndims, 0);
    int32_t step = -1;
    for (uint32_t i = 0; i < sizeof(uint64_t) * CHAR_BIT; ++i) {
        if (i % ndims == 0) ++step;
        if ((id >> i & 1) == 1) ccs[i % ndims] |= 1 << step;
    }
    return ccs;
}

uint64_t FortranOrderGenerator::getBlockCounter(std::vector<uint32_t> ccs, const std::vector<uint32_t> &dims) {
    uint64_t total_size = 1;
    uint64_t valor=0;

    for (int64_t i = ccs.size()-1; i>=0; i--){
        valor = ccs[i]*total_size+valor;
        total_size *= dims[i];
    }
    return valor;
}

std::vector<uint32_t> FortranOrderGenerator::getIndexes(uint64_t id, const std::vector<uint32_t> &dims) {
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

/* Number of elements to skip until the coordinate 'indexes' in matrix with 'dims' dimensions */
uint64_t FortranOrderGenerator::getIdFromIndexes(const std::vector<uint32_t> &dims, const std::vector<uint32_t> &indexes) {
    //std::cout<< " FortranOrderGenerator::getIdFromIndexes dims =";
    //for (uint64_t i = 0; i< dims.size(); i++) {
    //    std::cout<< dims[i] << ", ";
    //}
    //std::cout<<" idxs = ";
    //for (uint64_t i = 0; i< indexes.size(); i++) {
    //    std::cout<< indexes[i] << ", ";
    //}
    //std::cout<< std::endl;

    uint64_t id = *(indexes.begin());
    uint64_t accumulator = 1;
    for (uint64_t i = 0; i < (dims.size() - 1); i++) {
        accumulator *= dims[i];
        id += accumulator * indexes[i + 1];
    }
    //std::cout<< " FortranOrderGenerator::getIdFromIndexes offset ="<<id<<std::endl;
    return id;
}

/***
 * Given an array of 'dims' dimensions
 * @param elem_size       Size of every single element in the array
 * @param block_dims      Dimensions of the block being created
 * @param dims            Dimensions of the given array
 * @param data            Pointer to the beginning of the array data
 * @param output_data     Pointer to the beginning of the memory allocated for the block
 * @param output_data_end End of the memory allocated for the block
 */
void
FortranOrderGenerator::tessellate(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size,
                                 char *data,
                                 char *output_data, char *output_data_end) {

    //std::cout<< " FortranOrderGenerator::tessellate dims={";
    //for (uint32_t i = 0; i < dims.size(); i++) {
    //    std::cout<< dims[i]<<", ";
    //}
    //std::cout<< "} block_dims={";
    //for (uint32_t i = 0; i < block_dims.size(); i++) {
    //    std::cout<< block_dims[i]<<", ";
    //}
    //std::cout<< "} data="<<(void*)data<<std::endl;

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

bool FortranOrderGenerator::isDone() {
    done = (block_counter >= nblocks) || (cluster_counter >= nclusters);
    return done;
}


/* getBlockID - Returns the blockID from a coordinate
 * Args: cc : Coordinates in a blocked matrix
 */
uint32_t FortranOrderGenerator::getBlockID(std::vector<uint32_t> cc) {
    uint32_t block_id   = 0;
    for(uint32_t i = 0; i < blocks_dim.size()-1; i++) {
        uint32_t curr = cc[i];   // Current position
        uint32_t BITS = (uint32_t) log2(blocks_dim[i]) + 1; //Number of bits to codify current dimension value
        block_id <<= BITS;
        //std::cout<< "   cc["<<i<<"]="<<cc[i]<<" bits="<<BITS<<" block_id="<<block_id<<std::endl;
        block_id += curr;
    }
    return block_id;
}

/* getBlockID - Returns the blockID from a coordinate
 * Args: cc : Coordinates in a blocked matrix
 */
uint32_t FortranOrderGenerator::getClusterID(std::vector<uint32_t> cc) {
    return (uint32_t) (cc[blocks_dim.size()-1]);
}

Partition FortranOrderGenerator::getNextPartition() {
    //block_counter --> (block_id, cluster_id)

    char *output_data, *output_data_end;

    //std::cout<< "FortranOrderGenerator::getNextPartition block_counter " << block_counter << std::endl;
    // Calculate block coordinates
    std::vector<uint32_t> block_ccs = getIndexes(block_counter, blocks_dim);

    //Block parameters
    uint32_t cluster_id = getClusterID(block_ccs);

    // Block_id == block_ccs[0] | block_ccs[1] | ... | block_ccs[N-2]
    uint32_t block_id   = getBlockID(block_ccs);

    //std::cout<< " getNextPartition "<<nclusters<<"-> block_counter "<<block_counter<< "--> "<<block_id<< "."<<cluster_id<<std::endl;

    ++block_counter;

    if (data == NULL)
        return {cluster_id, block_id, nullptr};

    //Compute position in memory and chunks of data to copy

    //if any element of the block_ccs is equal to dim_split -> is a limit of the array -> recompute block size
    bool bound = false;
    for (uint32_t i = 0; i < ndims; ++i) {
        if (block_ccs[i] == blocks_dim[i] - 1) bound = true;
        block_ccs[i] *= row_elements;
    }

    //Number of bytes to skip until the coordinates
    uint64_t offset = getIdFromIndexes(metas.dims, block_ccs) * metas.elem_size;

    //Compute the real offset as: position inside the array * sizeof(element)
    char *input_start = ((char *) data) + offset;

    std::vector<uint32_t> dimsFortran = metas.dims;
    if (ndims>=2) { // Exchange last 2 dimensions
        auto tmp = dimsFortran[ndims-1];
        dimsFortran[ndims-1] = dimsFortran[ndims - 2];
        dimsFortran[ndims-2] = tmp;
    }
    if (!bound) {
        //In this case the block has size of row_elements in every_dimension
        //Create block
        output_data = (char *) malloc(block_size + sizeof(uint64_t));

        //copy the number of bytes
        memcpy(output_data, &block_size, sizeof(uint64_t));
        output_data += sizeof(uint64_t);
        output_data_end = output_data + block_size;
        //Copy the data
        tessellate(dimsFortran, block_dims, metas.elem_size, input_start, output_data, output_data_end);

    } else {
        //The block is a limit of the array, and its size needs to be recomputed and adjusted
        std::vector<uint32_t> bound_dims(ndims);
        //compute block size
        uint64_t bound_size = metas.elem_size;
        for (uint32_t i = 0; i < ndims; ++i) {
            //compute elem per dimension to be copied
            if (block_ccs[i] / row_elements != (blocks_dim[i] - 1)) {
                //Dimension isn't a limit, copy row_elements
                bound_dims[i] = row_elements;
            } else {
                //Is a limit, copy the remaining elements
                bound_dims[i] = (metas.dims[i] - (blocks_dim[i] - 1) * row_elements);
            }
            bound_size *= bound_dims[i];
        }

        if (bound_dims.size()>=2) { // bound_dims has been calculated as if the matrix was stored by rows --> exchange last 2 dimensions
            auto tmp = bound_dims[bound_dims.size()-1];
            bound_dims[bound_dims.size()-1] = bound_dims[bound_dims.size()-2];
            bound_dims[bound_dims.size()-2] = tmp;
        }

        //Create block
        output_data = (char *) malloc(bound_size + sizeof(uint64_t)); //chunk_size
        //Create block pointing to the memory

        memcpy(output_data, &bound_size, sizeof(uint64_t)); //copy the number of bytes
        output_data += sizeof(uint64_t);
        output_data_end = output_data + bound_size;
        //Copy the data
        tessellate(dimsFortran, bound_dims, metas.elem_size, input_start, output_data, output_data_end);
    }

    return {cluster_id, block_id, output_data - sizeof(uint64_t)};
}

int32_t FortranOrderGenerator::computeNextClusterId() {
    return cluster_counter ++;
}

/***
 * @param dims Dimensions of the future array
 * @param block_shape Dimensions of the block of data passed
 * @param elem_size Size of every single element in the array
 * @param output_array Pointer to the beginning of the position of the array where data should be written
 * @param input_block Pointer to the beginning of the the block
 * @param input_block_end End of the memory allocated for the block
 */
void FortranOrderGenerator::copy_block_to_array(std::vector<uint32_t> dims, std::vector<uint32_t> block_shape,
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


void FortranOrderGenerator::merge_partitions(const ArrayMetadata &metas, std::vector<Partition> chunks, void *data) {
    uint32_t ndims = (uint32_t) metas.dims.size();

    //std::cout << " Fortranorder:merge_partitions "<<chunks.size() << " chunks"<<std::endl;

    //Shape of the average block
    std::vector<uint32_t> block_shape(ndims, row_elements);

    //For each partition compute the future position inside the new array
    //Achieved using the cluster_id and block_id to recompute the ZorderId
    for (Partition chunk : chunks) {
        //Compute position in memory (Scale coordinates to element coordinates)
        std::vector<uint32_t> ccs = std::vector<uint32_t>(metas.dims);
        uint32_t block_id   = chunk.block_id;
        ccs[ccs.size()-1] = chunk.cluster_id;
        for(int32_t i = blocks_dim.size()-2; i >= 0; i--) {
            uint32_t BITS = (uint32_t) log2(blocks_dim[i]) + 1; //Number of bits to codify current dimension value
            uint32_t curr = block_id & ~(-1<< BITS);
            block_id >>= BITS;
            ccs[i] = curr;
            //std::cout<< "         ccs["<<i<<"]="<<ccs[i]<<" bits="<<BITS<<" block_id="<<block_id<<std::endl;
        }

        //if any element of the ccs is equal to dim_split -> is a limit of the array -> recompute chunk
        bool bound = false;
        //std::cout << "Blocks ";
        for (uint32_t i = 0; i < ndims; ++i) {
        //std::cout << "ccs["<<i<<"]="<<ccs[i]<<" blocks_dim["<<i<<"]="<<blocks_dim[i]<<std::endl;
            if (ccs[i] == blocks_dim[i] - 1) bound = true;
            ccs[i] *= row_elements;
        }


        //Number of elements to skip until the coordinates
        uint64_t offset = getIdFromIndexes(metas.dims, ccs);
        char *output_start = static_cast<char *>(data) + offset * metas.elem_size;
        char *input = (char *) chunk.data;
        uint64_t *retrieved_block_size = (uint64_t *) input;
        input += +sizeof(uint64_t);
        char *input_ends = input + *retrieved_block_size;
        //std::cout << " - bid:cid "<<chunk.block_id<<":"<<chunk.cluster_id<<" -> offset="<<offset<< " block_size="<<block_size<<" current block_size="<<*retrieved_block_size<<(bound?"BOUND":"")<<std::endl;

        std::vector<uint32_t> dimsFortran = metas.dims;
        if (ndims>=2) { // Exchange last 2 dimensions
            auto tmp = dimsFortran[ndims-1];
            dimsFortran[ndims-1] = dimsFortran[ndims - 2];
            dimsFortran[ndims-2] = tmp;
        }

        if (!bound) {

            if (*retrieved_block_size != block_size)
                throw ModuleException("Sth went wrong deciding "
                                      "the size of blocks while merging them into an array");


            copy_block_to_array(dimsFortran, block_shape, metas.elem_size, output_start, input, input_ends);


        } else {

            //The block is a limit of the array, and its size needs to be recomputed and adjusted
            std::vector<uint32_t> bound_dims(ndims);
            //compute block size
            uint64_t bound_size = metas.elem_size;
            for (uint32_t i = 0; i < ndims; ++i) {
                //compute elem per dimension to be copied
                if (ccs[i] / row_elements != (blocks_dim[i] - 1)) {
                    //Dimension isn't a limit, copy row_elements
                    bound_dims[i] = (int32_t) row_elements;
                } else {
                    //Is a limit, copy the remaining elements
                    bound_dims[i] = (uint32_t) (metas.dims[i] - (blocks_dim[i] - 1) * row_elements);
                }
                bound_size *= bound_dims[i];
            }
            if (bound_dims.size()>=2) {// bound_dims has been calculated as if the matrix was stored by rows --> exchange last 2 dimensions
                auto tmp = bound_dims[bound_dims.size()-1];
                bound_dims[bound_dims.size()-1] = bound_dims[bound_dims.size()-2];
                bound_dims[bound_dims.size()-2] = tmp;
            }

            //Copy the data
            copy_block_to_array(dimsFortran, bound_dims, metas.elem_size, output_start, input, input_ends);
        }
    }
}

PartitionIdxs FortranOrderGenerator::getNextPartitionIdxs() {
    // Calculate block coordinates
    std::vector<uint32_t> block_ccs = getIndexes(block_counter, blocks_dim);

    //Block parameters
    uint32_t cluster_id = (uint32_t) (block_ccs[blocks_dim.size()-1]);

    // Block_id == block_ccs[0] | block_ccs[1] | ... | block_ccs[N-2]
    uint32_t block_id   = 0;
    for(uint32_t i = 0; i < blocks_dim.size()-1; i++) {
        uint32_t curr = block_ccs[i];   // Current position
        uint32_t BITS = (uint32_t) log2(blocks_dim[i]) + 1; //Number of bits to codify current dimension value
        block_id <<= BITS;
        //std::cout<< "   block_ccs["<<i<<"]="<<block_ccs[i]<<" bits="<<BITS<<" block_id="<<block_id<<std::endl;
        block_id += curr;
    }
    block_counter ++;
    if (block_counter == nblocks) done = true;
    return {block_counter, cluster_id, block_id, block_ccs};
}

FortranOrderGeneratorFiltered::FortranOrderGeneratorFiltered(const ArrayMetadata &metas, void *data,
                                                           std::list<std::vector<uint32_t> > &coord)
        : FortranOrderGenerator(metas, data) {
    this->coord = coord;
}

bool FortranOrderGeneratorFiltered::isDone() {
    if (coord.empty()) done = true;
    return done;
}

int32_t FortranOrderGeneratorFiltered::computeNextClusterId() {
    std::vector<uint32_t> pos = coord.front();
    int32_t zorder = pos[pos.size()-1];
    coord.erase(coord.begin());
    return zorder;
}

Partition FortranOrderGeneratorFiltered::getNextPartition() {
    block_counter = getBlockCounter(coord.front(), blocks_dim);
    coord.erase(coord.begin());
    return FortranOrderGenerator::getNextPartition();
}
