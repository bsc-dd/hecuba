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


/*static PyObject* put_row(PyObject* self, PyObject* args);

static PyObject *get_row(PyObject* self, PyObject* args);

static PyObject* connectCassandra(PyObject* self, PyObject* args);


static PyObject* disconnectCassandra(PyObject* self);

static PyObject* addCacheTable(PyObject* self, PyObject* args);
*/
#endif //PREFETCHER_PREFETCHER_IMP_H
