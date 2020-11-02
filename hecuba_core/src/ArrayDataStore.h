#ifndef HFETCH_ARRAY_DATASTORE_H
#define HFETCH_ARRAY_DATASTORE_H


#include "SpaceFillingCurve.h"
#include "CacheTable.h"


class ArrayDataStore {

public:

    ArrayDataStore(const char *table, const char *keyspace, CassSession *session,
                   std::map<std::string, std::string> &config);

    ~ArrayDataStore();

    void store_numpy_into_cas_by_coords(const uint64_t *storage_id, ArrayMetadata &metadata, void *data,
                                        std::list<std::vector<uint32_t> > &coord) const;

    void store_numpy_into_cas(const uint64_t *storage_id, ArrayMetadata &metadata, void *data) const;

    void read_numpy_from_cas_by_coords(const uint64_t *storage_id, ArrayMetadata &metadata,
                                       std::list<std::vector<uint32_t> > &coord, void *save);

    void read_numpy_from_cas(const uint64_t *storage_id, ArrayMetadata &metadata, void *save);

    // Returns the metadata of the array identified by the storage_id
    ArrayMetadata *read_metadata(const uint64_t *storage_id) const;

    // Overwrite the metadata of the array identified by the given storage_id
    //void update_metadata(const uint64_t *storage_id, ArrayMetadata *metadata) const;

    //lgarrobe
    std::string TN  = "";
    void read_numpy_from_cas_arrow(const uint64_t *storage_id, ArrayMetadata &metadata, std::vector<uint64_t> &cols, void *save);
    void store_numpy_into_cas_as_arrow(const uint64_t *storage_id, ArrayMetadata &metadata,
                                       void *data) const;
    void store_numpy_into_cas_by_cols_as_arrow(const uint64_t *storage_id, ArrayMetadata &metadata, void *data, std::vector<uint32_t> &cols) const;

protected:

    void store_numpy_partition_into_cas(const uint64_t *storage_id , Partition part) const;


    CacheTable *cache = nullptr, *read_cache = nullptr;
    CacheTable *metadata_cache = nullptr, *metadata_read_cache=nullptr;

    SpaceFillingCurve partitioner;

    bool arrow_enabled = false;
    bool arrow_optane  = false; // Intel OPTANE disk enabled?
    std::string arrow_path  = "";
};


#endif //HFETCH_ArrayDataStore_H
