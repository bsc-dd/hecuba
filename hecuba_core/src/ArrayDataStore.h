#ifndef HFETCH_ARRAY_DATASTORE_H
#define HFETCH_ARRAY_DATASTORE_H


#include "SpaceFillingCurve.h"
#include "CacheTable.h"

#include <climits>
#include <list>
#include <set>

class ArrayDataStore {

public:

    ArrayDataStore(CacheTable *cache, CacheTable *read_cache,
                   std::map<std::string, std::string> &config, std::set<uint32_t> cluster_ids);

    ~ArrayDataStore();

    void store(const uint64_t *storage_id, ArrayMetadata *metadata, void *data) const;

    void *read(const uint64_t *storage_id, ArrayMetadata *metadata) const;

    // Returns the metadata of the array identified by the storage_id
    ArrayMetadata *read_metadata(const uint64_t *storage_id) const;

    // Overwrite the metadata of the array identified by the given storage_id
    void update_metadata(const uint64_t *storage_id, ArrayMetadata *metadata) const;

    void *read_n_coord(const uint64_t *storage_id, ArrayMetadata *metadata, std::list<std::vector<uint32_t> > crd,
                       void *save);


private:


    CacheTable *cache = nullptr, *read_cache = nullptr;

    SpaceFillingCurve *partitioner;

    std::set<uint32_t> cluster_ids = {};


};


#endif //HFETCH_ArrayDataStore_H
