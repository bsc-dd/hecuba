//
// Created by bscuser on 1/19/17.
//

#ifndef PREFETCHER_CACHE_TABLE_H
#define PREFETCHER_CACHE_TABLE_H

#include <iostream>
#include <python2.7/Python.h>
#include "Poco/LRUCache.h"
#include <cassandra.h>
#include <cstring>
#include <string>
#include <memory>


#include "TupleRow.h"
#include "TupleRowFactory.h"

class CacheTable {

public:

    CacheTable(uint32_t size, const char * table, const char * keyspace, const char* query, CassSession* session);

    ~CacheTable();

    TupleRowFactory* get_tuple_f();

    PyObject* get_row(PyObject *py_keys);

    int put_row(PyObject *row);

    int put_row(const CassRow *row);
private:


    /* CASSANDRA INFORMATION FOR RETRIEVING DATA */
    CassSession* session;
    const CassPrepared *prepared_query;

    //Key based on copy constructor, Value based on Poco:SharedPtr
    Poco::LRUCache<TupleRow, TupleRow> *myCache;

    TupleRowFactory* t_factory;

    void bind_keys(CassStatement *statement, TupleRow* keys);
};

#endif //PREFETCHER_CACHE_TABLE_H
