#ifndef HFETCH_ARRAYPARTITIONER_H
#define HFETCH_ARRAYPARTITIONER_H

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include "limits.h"

#define BLOCK_SIZE 4096
#define CLUSTER_SIZE 2

//Represents a block of data belonging to an array
struct Partition {
    Partition(uint32_t block, uint32_t cluster, void *chunk) {
        this->block_id = block;
        this->cluster_id = cluster;
        this->data = chunk;
    }

    uint32_t block_id;
    uint32_t cluster_id;
    void *data;
};

//TODO Inherit from CassUserType, pass the user type directly
//Represents the shape and type of an array
struct ArrayMetadata {
    std::vector<int32_t> dims;
    int32_t inner_type;
    uint32_t elem_size;
};


class SpaceFillingCurve {
public:
    virtual ~SpaceFillingCurve() {};

    virtual std::vector<Partition> make_partitions(const ArrayMetadata *metas, void *data) const;

    virtual void *merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const;

};


class ZorderCurve : public SpaceFillingCurve {

public:
    ~ZorderCurve() {};

    std::vector<Partition> make_partitions(const ArrayMetadata *metas, void *data) const;

    void *merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const;

    std::vector<uint32_t> getIndexes(uint64_t id, const std::vector<int32_t> &dims) const;

    std::vector<uint32_t> zorderInverse(uint64_t id, uint64_t ndims) const;

    uint64_t getIdFromIndexes(const std::vector<int32_t> &dims, const std::vector<uint32_t> &indexes) const;

    std::vector<uint32_t>
    scaleCoordinates(std::vector<int32_t> dims, uint64_t nblocks, std::vector<uint32_t> ccs) const;

};

#endif //HFETCH_ARRAYPARTITIONER_H
