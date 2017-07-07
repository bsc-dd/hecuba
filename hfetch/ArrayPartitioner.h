#ifndef HFETCH_ARRAYPARTITIONER_H
#define HFETCH_ARRAYPARTITIONER_H

#include <iostream>
#include <vector>
#include <cstring>

//Represents a block of data belonging to an array
struct Partition {
    Partition(int32_t block, int32_t cluster, const void *chunk) {
        this->block_id = block;
        this->cluster_id = cluster;
        this->data = chunk;
    }
    int32_t block_id;
    int32_t cluster_id;
    const void *data;
};
//TODO Inherit from CassUserType, pass the user type directly
//Represents the shape and type of an array
struct ArrayMetadata {
    std::vector<int32_t > dims;
    int32_t inner_type;
    uint64_t block_size;
    uint64_t cluster_size;

};



class ArrayPartitioner {
public:
    virtual ~ArrayPartitioner() {};

    virtual std::vector<Partition> make_partitions(const ArrayMetadata *metas, void* data) const;

    virtual void* merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const;

};


class ZorderPartitioner:public ArrayPartitioner {
    std::vector<Partition> make_partitions(const ArrayMetadata *metas, void* data) const;

    virtual void* merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) const;

};

#endif //HFETCH_ARRAYPARTITIONER_H
