#ifndef HFETCH_ARRAY_DATASTORE_H
#define HFETCH_ARRAY_DATASTORE_H


#include "SpaceFillingCurve.h"
#include "CacheTable.h"

#include <climits>
#include <list>
#include <set>

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

protected:

    CacheTable *cache = nullptr, *read_cache = nullptr;

    SpaceFillingCurve partitioner;

};


#endif //HFETCH_ArrayDataStore_H
