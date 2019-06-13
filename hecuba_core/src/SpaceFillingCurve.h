#ifndef HFETCH_SPACEFILLINGCURVE_H
#define HFETCH_SPACEFILLINGCURVE_H

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include "limits.h"

#define BLOCK_SIZE 4096
#define CLUSTER_SIZE 2
#define CLUSTER_END_FLAG INT_MAX-1
#define CLUSTER_ID_ARRAY INT_MAX

#define ZORDER_ALGORITHM 0
#define NO_PARTITIONS 1

//Represents a block of data belonging to an array
struct Partition {
    Partition(uint32_t cluster, uint32_t block, void *chunk) {
        this->block_id = block;
        this->cluster_id = cluster;
        this->data = chunk;
    }

    uint32_t cluster_id;
    uint32_t block_id;
    void *data;
};

//TODO Inherit from CassUserType, pass the user type directly
//Represents the shape and type of an array
struct ArrayMetadata {
    ArrayMetadata() {}

    ArrayMetadata(std::vector<uint32_t> dims, int32_t inner_type, uint32_t elem_size, uint8_t partition_type) {
        this->dims = dims;
        this->inner_type = inner_type;
        this->elem_size = elem_size;
        this->partition_type = partition_type;
    }

    std::vector<uint32_t> dims;
    int32_t inner_type;
    uint32_t elem_size;
    uint8_t partition_type;
};


class SpaceFillingCurve {
public:

    class PartitionGenerator {
    public:

        virtual ~PartitionGenerator() {};

        virtual bool isDone() = 0;

        virtual Partition getNextPartition() = 0;

        virtual int32_t computeNextClusterId() = 0;

        virtual void simpleNextClusterId() = 0;

        virtual void *merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks) = 0;

    };


    ~SpaceFillingCurve() {};

    static PartitionGenerator *make_partitions_generator(const ArrayMetadata *metas, void *data);

protected:

    class SpaceFillingGenerator : public PartitionGenerator {
    public:
        SpaceFillingGenerator();

        SpaceFillingGenerator(const ArrayMetadata *metas, void *data);

        Partition getNextPartition();

        int32_t computeNextClusterId();

        void simpleNextClusterId();

        bool isDone() { return done; };

        void *merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks);

    protected:
        bool done;
        const ArrayMetadata *metas;
        void *data;
        uint64_t total_size;
    };

};


class ZorderCurveGenerator : public SpaceFillingCurve::PartitionGenerator {
public:
    ZorderCurveGenerator();

    ZorderCurveGenerator(const ArrayMetadata *metas, void *data);

    Partition getNextPartition();

    int32_t computeNextClusterId();

    void simpleNextClusterId();

    bool isDone() {
        if (block_counter >= nblocks) done = true;
        return done;
    };

    uint64_t computeZorder(std::vector<uint32_t> cc);

    std::vector<uint32_t> zorderInverse(uint64_t id, uint64_t ndims);

    std::vector<uint32_t> getIndexes(uint64_t id, const std::vector<uint32_t> &dims);

    uint64_t getIdFromIndexes(const std::vector<uint32_t> &dims, const std::vector<uint32_t> &indexes);

    void *merge_partitions(const ArrayMetadata *metas, std::vector<Partition> chunks);

private:
    bool done;
    const ArrayMetadata *metas;
    void *data;
    uint32_t ndims, row_elements;
    uint64_t block_size, nblocks;
    std::vector<uint32_t> block_dims, blocks_dim, bound_dims;
    uint64_t block_counter;


    static void tessellate(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size, char *data,
                           char *output_data, char *output_data_end);

    static void
    copy_block_to_array(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size, char *data,
                        char *output_data, char *output_data_end);

};

#endif //HFETCH_SPACEFILLINGCURVE_H
