#ifndef PREFETCHER_CACHE_TABLE_H
#define PREFETCHER_CACHE_TABLE_H

#include <iostream>
#include <cassandra.h>
#include <cstring>
#include <string>
#include <memory>

#include "Poco/LRUCache.h"

#include "TupleRow.h"
#include "TupleRowFactory.h"
#include "Writer.h"


class CacheTable {

public:
    CacheTable(const TableMetadata *table_meta,
               CassSession *session,
               std::map<std::string, std::string> &config);

    ~CacheTable();


    /** C++ METHODS **/

    void put_crow(void *keys, void *values);

    std::vector<const TupleRow *> get_crow(const TupleRow *py_keys);


    std::vector<const TupleRow *> get_crow(void *keys);

    void add_to_cache(void *keys, void *values);

    void put_crow(const TupleRow *row);

    void put_crow(const TupleRow *keys, const TupleRow *values);

    const TableMetadata *get_metadata() {
        return table_metadata;
    }

private:

    std::vector<const TupleRow *> retrieve_from_cassandra(const TupleRow *keys);

    /* CASSANDRA INFORMATION FOR RETRIEVING DATA */
    CassSession *session;
    const CassPrepared *prepared_query;

    //Key based on copy constructor, Value based on Poco:SharedPtr
    Poco::LRUCache<TupleRow, TupleRow> *myCache;

    TupleRowFactory *keys_factory;
    TupleRowFactory *values_factory;

    const TableMetadata *table_metadata;

    Writer *writer;

};

#endif //PREFETCHER_CACHE_TABLE_H
