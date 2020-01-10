#ifndef HFETCH_ARRAY_DATASTORE_H
#define HFETCH_ARRAY_DATASTORE_H


#include "SpaceFillingCurve.h"
#include "CacheTable.h"

#include <climits>


class ArrayDataStore {

public:

    ArrayDataStore(const char *table, const char *keyspace, CassSession *session,
                   std::map<std::string, std::string> &config);

    ~ArrayDataStore();

    void store(const uint64_t *storage_id, ArrayMetadata &metadata, void *data) const;

    void *read(const uint64_t *storage_id, ArrayMetadata &metadata) const;

protected:


    CacheTable *cache = nullptr, *read_cache = nullptr;

    SpaceFillingCurve partitioner;

};


#endif //HFETCH_ArrayDataStore_H
