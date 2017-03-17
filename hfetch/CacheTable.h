//
// Created by bscuser on 1/19/17.
//

#ifndef PREFETCHER_CACHE_TABLE_H
#define PREFETCHER_CACHE_TABLE_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <python2.7/Python.h>
#include "Poco/LRUCache.h"
#include <cassandra.h>
#include <cstring>
#include <string>
#include <memory>


#include "TupleRow.h"
#include "TupleRowFactory.h"

#include "Prefetch.h"
#include "Writer.h"


class CacheTable {

public:
    CacheTable(const std::string &table,const std::string &keyspace,
               const std::vector<std::string> &keyn,
               const std::vector< std::vector<std::string>>  &columns_n ,
               const std::string &token_range_pred,
               const std::vector<std::pair<int64_t, int64_t>> &tkns,
               CassSession *session,
               std::map<std::string,std::string> &config);

    ~CacheTable();


    PyObject *get_row(PyObject *py_keys);

    void put_row(PyObject *key, PyObject *value);


    Prefetch* get_keys_iter(uint32_t prefetch_size);

    Prefetch* get_values_iter(uint32_t prefetch_size);

    Prefetch* get_items_iter(uint32_t prefetch_size);
    const TupleRow *get_crow(TupleRow *py_keys);

    TupleRowFactory* _test_get_keys_factory(){ return keys_factory;}
    TupleRowFactory* _test_get_value_factory(){ return values_factory;}
    TupleRowFactory* _test_get_items_factory(){ return items_factory;}

private:

    std::string select_keys;
    std::string select_values;
    std::string select_all;
    std::string token_predicate;
    std::string get_predicate;
    std::string cache_query;

    std::vector<std::string> key_names;
    std::vector< std::vector<std::string>> columns_names;
    std::vector< std::vector<std::string>> all_names;
    std::vector<std::pair<int64_t, int64_t>> tokens;


    /* CASSANDRA INFORMATION FOR RETRIEVING DATA */
    CassSession *session;
    const CassPrepared *prepared_query;

    //Key based on copy constructor, Value based on Poco:SharedPtr
    Poco::LRUCache<TupleRow, TupleRow> *myCache;

    TupleRowFactory *keys_factory;
    TupleRowFactory *values_factory;
    TupleRowFactory *items_factory;

    Writer *writer;

};

#endif //PREFETCHER_CACHE_TABLE_H
