//
// Created by bscuser on 1/19/17.
//

#ifndef PREFETCHER_PREFETCHER_IMP_H
#define PREFETCHER_PREFETCHER_IMP_H

#include "CacheTable.h"
#include <tuple>
#include <iostream>
#include <vector>
#include <unordered_map>
#include "StorageInterface.h"
#include "Poco/LRUCache.h"
#include <cassandra.h>
#include "python2.7/Python.h"
#include "structmember.h"
#include "Prefetch.h"
#include "PythonParser.h"
#include <map>

StorageInterface* storage;
PythonParser* parser;


typedef struct {
    PyObject_HEAD
    CacheTable *T;
    const TableMetadata* metadata;
    std::vector<std::pair<int64_t, int64_t>> token_ranges;
} HCache;


typedef struct {
    PyObject_HEAD
    Prefetch *P;
    CacheTable *baseTable;
    const TableMetadata* metadata;
    std::vector<std::pair<int64_t, int64_t>> token_ranges;
    bool update_cache;
} HIterator;


typedef struct {
    PyObject_HEAD
    Writer *W;
    CacheTable *baseTable;
    const TableMetadata* metadata;
} HWriter;

static PyObject* create_iter_keys(HCache *self, PyObject* args);
static PyObject* create_iter_values(HCache *self, PyObject* args);
static PyObject* create_iter_items(HCache *self, PyObject* args);


#endif //PREFETCHER_PREFETCHER_IMP_H
