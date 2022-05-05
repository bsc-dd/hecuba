#ifndef HFETCH_SPACEFILLINGCURVE_H
#define HFETCH_SPACEFILLINGCURVE_H

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include "limits.h"
#include <map>
#include <list>

#define BLOCK_SIZE 4096
#define CLUSTER_SIZE 2
#define CLUSTER_END_FLAG INT_MAX-1
#define CLUSTER_ID_ARRAY INT_MAX

#define ZORDER_ALGORITHM 0
#define NO_PARTITIONS 1
#define COLUMNAR 2
#define FORTRANORDER 3

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

    bool operator<(const Partition &part) const { return cluster_id < part.cluster_id; };

};

//Represents the indexes of a block of data belonging to an array (a Partition)
struct PartitionIdxs {
    PartitionIdxs(uint64_t id, uint32_t cluster, uint32_t block, std::vector<uint32_t> ccs) {
        this->id         = id;
        this->cluster_id = cluster;
        this->block_id   = block;
        this->ccs        = ccs;
    }

    uint64_t              id;         // The Zorder_id of the block (0..num_blocks)
    uint32_t              cluster_id; // (derived from 'id') Zorder cluster_id
    uint32_t              block_id;   // (derived from 'id') Zorder block_id
    std::vector<uint32_t> ccs;        // (derived form 'id' and the array dimensions) The indexes of the blocks at each dimension
};

//TODO Inherit from CassUserType, pass the user type directly
//Represents the shape and type of an array
struct ArrayMetadata {
    ArrayMetadata() = default;

    uint32_t flags=0;
    uint32_t elem_size=0;
    uint8_t partition_type = ZORDER_ALGORITHM;
    char typekind = ' ';
    char  byteorder = ' ';
    std::vector<uint32_t> dims;
    std::vector<uint32_t> strides;
    //int32_t inner_type = 0;
    uint64_t get_array_size() {
        uint64_t size = 1;
        for (auto i: dims) {
            size *= i;
        }
        size *= elem_size;
        return size;
    }
};


class SpaceFillingCurve {
public:

    class PartitionGenerator {
    public:

        virtual ~PartitionGenerator() {};

        virtual bool isDone() = 0;

        virtual Partition getNextPartition() = 0;

        virtual int32_t computeNextClusterId() = 0;

        virtual PartitionIdxs getNextPartitionIdxs()  = 0;

        virtual uint32_t getBlockID(std::vector<uint32_t> cc) = 0;
        virtual uint32_t getClusterID(std::vector<uint32_t> cc) = 0;

        virtual void merge_partitions(const ArrayMetadata &metas, std::vector<Partition> chunks, void *data) = 0;
    };

    ~SpaceFillingCurve() {};

    static PartitionGenerator *
    make_partitions_generator(const ArrayMetadata &metas, void *data);

    static PartitionGenerator *make_partitions_generator(const ArrayMetadata &metas, void *data,
                                                         std::list<std::vector<uint32_t> > &coord);


protected:

    class SpaceFillingGenerator : public PartitionGenerator {
    public:
        SpaceFillingGenerator();

        SpaceFillingGenerator(const ArrayMetadata &metas, void *data);

        Partition getNextPartition() override;

        int32_t computeNextClusterId() override;

        PartitionIdxs getNextPartitionIdxs() override;

        uint32_t getBlockID(std::vector<uint32_t> cc) override;
        uint32_t getClusterID(std::vector<uint32_t> cc) override;

        bool isDone() override { return done; };

        void merge_partitions(const ArrayMetadata &metas, std::vector<Partition> chunks, void *data) override;

    protected:
        bool done;
        const ArrayMetadata metas;
        void *data;
        uint64_t total_size;
    };

};


class ZorderCurveGenerator : public SpaceFillingCurve::PartitionGenerator {
public:

    ZorderCurveGenerator();

    ZorderCurveGenerator(const ArrayMetadata &metas, void *data);

    Partition getNextPartition() override;

    int32_t computeNextClusterId() override;

    PartitionIdxs getNextPartitionIdxs() override;

    uint32_t getBlockID(std::vector<uint32_t> cc) override;
    uint32_t getClusterID(std::vector<uint32_t> cc) override;

    bool isDone() override {
        if (block_counter >= nblocks) done = true;
        return done;
    };

    uint64_t computeZorder(std::vector<uint32_t> cc);
    uint64_t getBlockCounter(std::vector<uint32_t> ccs, const std::vector<uint32_t> &dims);

    std::vector<uint32_t> zorderInverse(uint64_t id, uint64_t ndims);

    std::vector<uint32_t> getIndexes(uint64_t id, const std::vector<uint32_t> &dims);

    uint64_t getIdFromIndexes(const std::vector<uint32_t> &dims, const std::vector<uint32_t> &indexes);

    void merge_partitions(const ArrayMetadata &metas, std::vector<Partition> chunks, void *data) override;

private:
    bool done;
    const ArrayMetadata metas;
    void *data;
    uint32_t ndims, row_elements, nreddims;
    uint64_t block_size, nblocks, nclusters;
    std::vector<uint32_t> block_dims, bound_dims, clusters_dim;


    static void tessellate(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size, char *data,
                           char *output_data, char *output_data_end);

    static void
    copy_block_to_array(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size, char *data,
                        char *output_data, char *output_data_end);

protected:
    uint64_t block_counter, cluster_counter;
    std::vector<uint32_t> blocks_dim;
};


class ZorderCurveGeneratorFiltered : public ZorderCurveGenerator {
public:

    ZorderCurveGeneratorFiltered(const ArrayMetadata &metas, void *data, std::list<std::vector<uint32_t> > &coord);

    int32_t computeNextClusterId() override;

    Partition getNextPartition() override;

    bool isDone() override;

private:
    std::list<std::vector<uint32_t> > coord;
    bool done = false;
};


class FortranOrderGenerator : public SpaceFillingCurve::PartitionGenerator {
public:

    FortranOrderGenerator();

    FortranOrderGenerator(const ArrayMetadata &metas, void *data);

    Partition getNextPartition() override;

    int32_t computeNextClusterId() override;

    PartitionIdxs getNextPartitionIdxs() override;

    bool isDone() override;

    uint64_t computeZorder(std::vector<uint32_t> cc);

    uint64_t getBlockCounter(std::vector<uint32_t> ccs, const std::vector<uint32_t> &dims);

    std::vector<uint32_t> zorderInverse(uint64_t id, uint64_t ndims);

    std::vector<uint32_t> getIndexes(uint64_t id, const std::vector<uint32_t> &dims);

    uint64_t getIdFromIndexes(const std::vector<uint32_t> &dims, const std::vector<uint32_t> &indexes);

    uint32_t getBlockID(std::vector<uint32_t> cc) override;
    uint32_t getClusterID(std::vector<uint32_t> cc) override;

    void merge_partitions(const ArrayMetadata &metas, std::vector<Partition> chunks, void *data) override;


private:
    bool done;
    const ArrayMetadata metas;
    void *data;
    uint32_t ndims, row_elements, nreddims;
    //uint64_t block_size, nblocks, nclusters;
    //std::vector<uint32_t> block_dims, blocks_dim, bound_dims, clusters_dim;
    uint64_t block_size;
    uint64_t nblocks;   // Total number of blocks
    uint64_t nclusters; // Total number of clusters
    std::vector<uint32_t> clusters_dim; // Num clusters per dimension (half the number of blocks for the first 2 dims)
    std::vector<uint32_t> block_dims;   // ???? 
    std::vector<uint32_t> bound_dims;   // ????


    static void tessellate(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size, char *data,
                           char *output_data, char *output_data_end);

    static void
    copy_block_to_array(std::vector<uint32_t> dims, std::vector<uint32_t> block_dims, uint32_t elem_size, char *data,
                        char *output_data, char *output_data_end);


protected:
    uint64_t block_counter, cluster_counter;
    std::vector<uint32_t> blocks_dim;   // Num blocks per dimension
};
class FortranOrderGeneratorFiltered : public FortranOrderGenerator {
public:

    FortranOrderGeneratorFiltered(const ArrayMetadata &metas, void *data, std::list<std::vector<uint32_t> > &coord);

    int32_t computeNextClusterId() override;

    Partition getNextPartition() override;

    bool isDone() override;

private:
    std::list<std::vector<uint32_t> > coord;
    bool done = false;
};
#endif //HFETCH_SPACEFILLINGCURVE_H
