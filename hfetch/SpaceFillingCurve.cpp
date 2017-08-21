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

std::vector<uint32_t>  ZorderCurve::zorderInverse(uint64_t id, uint64_t ndims) const{
    std::vector<uint32_t> ccs = std::vector<uint32_t>(ndims,0);
    int32_t step = -1;
    for (uint32_t i = 0; i<sizeof(uint64_t)*CHAR_BIT; ++i){
        if (i%ndims == 0) ++step;
        if ((id >> i &1) == 1) ccs[i%ndims]|=1<<step;
    }
    return ccs;
}

std::vector<uint32_t> ZorderCurve::getIndexes(uint64_t id,const std::vector<int32_t > &dims) const {
    uint64_t total_size=1;
    for (int32_t dim: dims) {
        total_size *= dim;
    }
    total_size/=dims[0];

    std::vector<uint32_t > ccs = std::vector<uint32_t >(dims.size());
    uint32_t i = 0;
    for (; i<ccs.size()-1; ++i) {
        std::ldiv_t dv = std::div((int64_t)id,total_size);
        ccs[i]=(uint32_t)dv.quot;
        total_size/=dims[i+1];
        id = (uint64_t) dv.rem;
    }
    ccs[i]=(uint32_t) id;
    return ccs;
}

uint64_t ZorderCurve::getIdFromIndexes(const std::vector<int32_t > &dims,const std::vector<uint32_t > &indexes) const {
    uint64_t id = *(--indexes.end());
    uint64_t accumulator = 1;
    for (int32_t i = dims.size() -1; i>0 ; --i) {
        accumulator*=dims[i];
        id+=accumulator*indexes[i-1];
    }
    return id;
}

std::vector<uint32_t > ZorderCurve::scaleCoordinates(std::vector<int32_t> dims, uint64_t nblocks, std::vector<uint32_t> ccs) const {
    std::vector<uint32_t > block_dim_elem = std::vector<uint32_t >(dims.size());
    uint64_t product_dims = 1;
    for (int32_t i = 0; i< block_dim_elem.size(); ++i) {
        product_dims *= dims[i];
    }
    for (int32_t i = 0; i< block_dim_elem.size(); ++i) {
        uint64_t nblocks_d  = (nblocks*BLOCK_SIZE)/product_dims;//TODO this is equal to size of element
        block_dim_elem[i] = ccs[i]*dims[i]/nblocks_d;
    }
    return block_dim_elem;
}



std::vector<Partition> ZorderCurve::make_partitions(const ArrayMetadata *metas, void *data) const {
    uint32_t ndims = (uint32_t) metas->dims.size();

    uint64_t dims_product = 1;
    for (uint64_t i = 0; i< metas->dims.size() - 1; ++i) {
        dims_product*=metas->dims[i];
    }

    uint64_t total_size = metas->elem_size;
    for (int32_t dim: metas->dims) {
        total_size *= dim;
    }
    char* input_end = ((char*) data)+total_size;
    char* output_data, *output_data_end;
    // Compute the best fitting block
    uint64_t b = BLOCK_SIZE;
    if (total_size<b) b = total_size;
    uint64_t block_size = b - (b%metas->elem_size);
    uint64_t row_elements=(uint64_t) std::floor(pow(block_size/metas->elem_size,(1.0/metas->dims.size())));
    block_size = (uint64_t) pow(row_elements,ndims) * metas->elem_size;

    //Compute the number of blocks
    uint64_t min_nblocks = (uint64_t) std::ceil(total_size/(double)block_size);
    uint64_t dim_split = (uint64_t) std::ceil(pow((min_nblocks),(1.0/metas->dims.size())));
    uint64_t nblocks = (uint64_t) pow(dim_split,metas->dims.size());

    //Create the blocks
    std::vector<Partition> parts = std::vector<Partition>(nblocks,{0,0, nullptr});

    //Compute offsets to copy data
    uint64_t row_elements_size = row_elements * metas->elem_size;
    uint64_t row_offset = dims_product * metas->elem_size;

    //Fill them with the data
    for (uint64_t block_i = 0; block_i < nblocks; ++block_i) {
        //Block parameters
        parts[block_i].cluster_id = (uint32_t) (block_i >> CLUSTER_SIZE);
        int64_t mask = -1 << (sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE);
        mask = (uint64_t) mask >> sizeof(uint64_t) * CHAR_BIT - CLUSTER_SIZE;
        parts[block_i].block_id = (uint32_t) (block_i & mask);

        //Compute position in memory and chunks of data to copy
        std::vector<uint32_t> ccs = zorderInverse(block_i, metas->dims.size()); //Block coordinates
        //if any element of the ccs is equal to dim_split -> is a limit of the array -> recompute chunk
        /* TODO The problem here:
         * some blocks are not present, for an instance: 3x3 array doesn't have a block with
         * block_id = 5; however it has block_id=0,1,2,3,4,6,8,9,12
         */

        /*TODO When a block is a bound of the array it must have less elements
         * than the other blocks
         */
        for (uint32_t i =0; i<ccs.size(); ++i) {
            uint64_t factor_dim_i = metas->dims[i]/dim_split;
            ccs[i]*=factor_dim_i;
        }
        uint64_t offset = getIdFromIndexes(metas->dims,ccs);

        //Compute the real offset as: position inside the array * sizeof(singleelement)
        char* input_start = ((char*)data)+offset*metas->elem_size;

        //Create block
        output_data = (char *) malloc(block_size+sizeof(uint64_t)); //chunk_size
        //Create block pointing to the memory
        parts[block_i].data=output_data;
        memcpy(output_data,&block_size,sizeof(uint64_t)); //copy the number of bytes
        output_data+=sizeof(uint64_t);
        output_data_end = output_data+block_size;
        while (output_data<output_data_end) {
            //last_dim*metas->elem_size-(block_size/product_dims*metas->elem_size);
            if (input_start+row_elements_size>input_end) {
                std::cout << "Overflow" << std::endl;
                break;
            }
            memcpy(output_data, input_start, row_elements_size);
            input_start += row_offset;
            output_data += row_elements_size;
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


