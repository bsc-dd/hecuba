#ifndef HFETCH_ARRAYPARTITIONER_H
#define HFETCH_ARRAYPARTITIONER_H

#include <iostream>
#include <vector>
#include <cstring>

//Represents a block of data belonging to an array
struct Partition {
    Partition(int32_t block, int32_t cluster, void *chunk) {
        this->block_id = block;
        this->cluster_id = cluster;
        this->data = chunk;
    }
    int32_t block_id;
    int32_t cluster_id;
    void *data;
};
//TODO Inherit from CassUserType, pass the user type directly
//Represents the shape and type of an array
struct ArrayMetadata {
    std::vector<int32_t > dims;
    int32_t inner_type;
};



class ArrayPartitioner {
public:
    virtual ~ArrayPartitioner() {};

    virtual std::vector<Partition> make_partitions(ArrayMetadata *metas, void* data) const;
};


class ZorderPartitioner:public ArrayPartitioner {
    std::vector<Partition> make_partitions(ArrayMetadata *metas, void* data) const;
};

#endif //HFETCH_ARRAYPARTITIONER_H
