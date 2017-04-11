#ifndef PREFETCHER_PREFETCHER_IMP_H
#define PREFETCHER_PREFETCHER_IMP_H

#include <python2.7/Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include "numpy/arrayobject.h"
//#include "structmember.h"


#include <tuple>
#include <iostream>
#include <vector>

#include <map>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <stdlib.h>
#include <cassert>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <cassandra.h>
#include "ModuleException.h"
#include "TupleRow.h"
#include "Prefetch.h"
#include "PythonParser.h"
#include "CacheTable.h"
#include "StorageInterface.h"


StorageInterface* storage;
PythonParser* parser;


typedef struct {
    PyObject_HEAD
    CacheTable *T;
    const TableMetadata* metadata;
    std::vector<std::pair<int64_t, int64_t>> token_ranges;
    bool has_numpy;
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
    bool has_numpy;
} HWriter;

static PyObject* create_iter_keys(HCache *self, PyObject* args);
static PyObject* create_iter_values(HCache *self, PyObject* args);
static PyObject* create_iter_items(HCache *self, PyObject* args);




void print_list_refc(PyObject* list) {
    std::cout << " >> Check list refs " << Py_REFCNT(list) << std::endl;
    for (uint16_t i= 0; i<PyList_GET_SIZE(list); ++i) {
        std::cout << "Element: " << i << " has REFS: " << Py_REFCNT(PyList_GetItem(list,i)) << std::endl;
    }
    std::cout << " >> Done check list "<< std::endl;
}


#endif //PREFETCHER_PREFETCHER_IMP_H
