#ifndef PREFETCHER_PREFETCHER_IMP_H
#define PREFETCHER_PREFETCHER_IMP_H

#define PY_SSIZE_T_CLEAN

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#if __cplusplus <= 199711L
#error This library needs at least a C++11 compliant compiler
#endif

#include "numpy/arrayobject.h"
#include <structmember.h>


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
#include "../ModuleException.h"
#include "../TupleRow.h"
#include "../Prefetch.h"
#include "PythonParser.h"
#include "../CacheTable.h"
#include "../StorageInterface.h"


std::shared_ptr<StorageInterface> storage; //StorageInterface* storage;

typedef struct {
    PyObject_HEAD
    CacheTable *T;
    std::vector<std::pair<int64_t, int64_t>> token_ranges;
    PythonParser *keysParser, *valuesParser;
} HCache;


typedef struct {
    PyObject_HEAD
    Prefetch *P;
    std::vector<std::pair<int64_t, int64_t>> token_ranges;
    PythonParser *rowParser;
} HIterator;


typedef struct {
    PyObject_HEAD
    Writer *W;
    PythonParser *keysParser, *valuesParser;
} HWriter;


typedef struct {
    PyObject_HEAD
    NumpyStorage *NumpyDataStore;
} HNumpyStore;

typedef struct {
    PyObject_HEAD
    ArrayMetadata np_metas;
} HArrayMetadata;


static PyObject *create_iter_keys(HCache *self, PyObject *args);

static PyObject *create_iter_values(HCache *self, PyObject *args);

static PyObject *create_iter_items(HCache *self, PyObject *args);


void print_list_refc(PyObject *list) {
    std::cout << " >> Check list refs " << Py_REFCNT(list) << std::endl;
    for (uint16_t i = 0; i < PyList_GET_SIZE(list); ++i) {
        std::cout << "Element: " << i << " has REFS: " << Py_REFCNT(PyList_GetItem(list, i)) << std::endl;
    }
    std::cout << " >> Done check list " << std::endl;
}


#endif //PREFETCHER_PREFETCHER_IMP_H
