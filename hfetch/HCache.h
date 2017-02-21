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

#include "Poco/LRUCache.h"
#include <cassandra.h>
#include "python2.7/Python.h"
#include "structmember.h"
#include "Prefetch.h"



CassSession *session = NULL;
CassCluster *cluster = NULL;

typedef struct {
    PyObject_HEAD
    CacheTable *T;
} HCache;


typedef struct {
    PyObject_HEAD
    Prefetch *P;
} HIterator;

static PyObject* create_iter_keys(HCache *self, PyObject* args);
static PyObject* create_iter_items(HCache *self, PyObject* args);
static PyObject* create_iter_values(HCache *self, PyObject* args);


#endif //PREFETCHER_PREFETCHER_IMP_H
