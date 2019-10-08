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
                   std::map<std::string, std::string> &config);

    ~ArrayDataStore();

    // Returns the metadata of the array identified by the storage_id
    ArrayMetadata *read_metadata(const uint64_t *storage_id) const;

    // Overwrite the metadata of the array identified by the given storage_id
    void update_metadata(const uint64_t *storage_id, ArrayMetadata *metadata) const;

    void store_numpy_into_cas(const uint64_t *storage_id, ArrayMetadata *metadata, void *data, uint64_t size_list) const;

    void *read_numpy_from_cas(const uint64_t *storage_id, ArrayMetadata *metadata, std::list<std::vector<uint32_t> > &crd,
                       void *save);


private:


    CacheTable *cache = nullptr, *read_cache = nullptr;

    SpaceFillingCurve partitioner;

};


#endif //HFETCH_ArrayDataStore_H
