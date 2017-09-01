#ifndef HFETCH_SPACEFILLINGCURVE_H
#define HFETCH_SPACEFILLINGCURVE_H

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
    std::vector<uint32_t> dims;
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

    uint64_t computeZorder(std::vector<uint32_t> cc) const;

    std::vector<uint32_t> zorderInverse(uint64_t id, uint64_t ndims) const;

    std::vector<uint32_t> getIndexes(uint64_t id, const std::vector<uint32_t> &dims) const;

    uint64_t getIdFromIndexes(const std::vector<uint32_t> &dims, const std::vector<uint32_t> &indexes) const;


private:

    void tessellate(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size, char *data,
                    char *output_data, char *output_data_end) const;

    void copy_block_to_array(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size, char *data,
                             char *output_data, char *output_data_end) const;


};

#endif //HFETCH_SPACEFILLINGCURVE_H
