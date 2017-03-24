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



class CacheTable {

public:
    CacheTable(const TableMetadata* table_meta,
               CassSession *session,
               std::map<std::string,std::string> &config);

    ~CacheTable();


    /** C++ METHODS **/

    void put_crow(void* keys,void* values);

    const TupleRow *get_crow(TupleRow *py_keys);

    std::shared_ptr<void> get_crow(void* keys);

    /** TESTING METHODS **/

    TupleRowFactory* _test_get_keys_factory(){ return keys_factory;}
    TupleRowFactory* _test_get_value_factory(){ return values_factory;}

private:

    const TupleRow* retrieve_from_cassandra(TupleRow *keys);

    /* CASSANDRA INFORMATION FOR RETRIEVING DATA */
    CassSession *session;
    const CassPrepared *prepared_query;

    //Key based on copy constructor, Value based on Poco:SharedPtr
    Poco::LRUCache<TupleRow, TupleRow> *myCache;

    TupleRowFactory *keys_factory;
    TupleRowFactory *values_factory;

};

#endif //PREFETCHER_CACHE_TABLE_H
