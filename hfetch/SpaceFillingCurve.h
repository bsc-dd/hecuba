#ifndef HFETCH_ARRAYPARTITIONER_H
#define HFETCH_ARRAYPARTITIONER_H

#include <iostream>
#include <vector>
#include <cstring>
#include "limits.h"

#define BLOCK_SIZE 16

//Represents a block of data belonging to an array
struct Partition {
    Partition(uint32_t block, uint32_t cluster, const void *chunk) {
        this->block_id = block;
        this->cluster_id = cluster;
        this->data = chunk;
    }

    uint32_t block_id;
    uint32_t cluster_id;
    const void *data;
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

};

#endif //HFETCH_ARRAYPARTITIONER_H
