#ifndef PREFETCHER_CACHE_TABLE_H
#define PREFETCHER_CACHE_TABLE_H

#include <iostream>
#include <cassandra.h>
#include <cstring>
#include <string>
#include <memory>

#include "TimestampGenerator.h"
#include "TupleRow.h"
#include "TupleRowFactory.h"
#include "KVCache.h"
#include "Writer.h"


class CacheTable {

public:
    CacheTable(const TableMetadata *table_meta,
               CassSession *session,
               std::map<std::string, std::string> &config);

    ~CacheTable();

    const TableMetadata *get_metadata() {
        return table_metadata;
    }

    const void flush_elements() const;
    const void wait_elements() const ;

    /*** TupleRow ops ***/

    std::vector<const TupleRow *> get_crow(const TupleRow *py_keys);

    std::vector<const TupleRow *> get_crow(void *keys);

    void put_crow(const TupleRow *keys, const TupleRow *values);

    void delete_crow(const TupleRow *keys);

    /*** Raw pointers ops ***/

    void put_crow(void *keys, void *values);

    void add_to_cache(void *keys, void *values);

private:

    std::vector<const TupleRow *> retrieve_from_cassandra(const TupleRow *keys);

    /* CASSANDRA INFORMATION FOR RETRIEVING DATA */
    CassSession *session;
    const CassPrepared *prepared_query, *delete_query;

    bool disable_timestamps;
    TimestampGenerator *timestamp_gen;

    //Key and Value copy constructed
    KVCache<TupleRow, TupleRow> *myCache;

    TupleRowFactory *keys_factory;
    TupleRowFactory *values_factory;

    const TableMetadata *table_metadata;

    Writer *writer;

};

#endif //PREFETCHER_CACHE_TABLE_H
