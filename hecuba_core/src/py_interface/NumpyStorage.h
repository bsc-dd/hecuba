#ifndef HFETCH_NUMPYSTORAGE_H
#define HFETCH_NUMPYSTORAGE_H

#include "../ArrayDataStore.h"

#include <Python.h>
#include <climits>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include "numpy/arrayobject.h"
#include "UnitParser.h"

/***
 * Responsible to store a numpy to the keyspace.table_numpies, associating an attribute_name and a storage_id(uuid)
 */

class NumpyStorage : public ArrayDataStore {

public:

    NumpyStorage(const char *table, const char *keyspace, CassSession *session,
                 std::map<std::string, std::string> &config);

    ~NumpyStorage();

    void store_numpy(const uint64_t *storage_id, PyArrayObject *numpy, ArrayMetadata &) const;

    PyObject *read_numpy(const uint64_t *storage_id, ArrayMetadata &np_metas);

    ArrayMetadata make_metadata(PyObject *py_np_metas) const;
};


#endif //HFETCH_NUMPYSTORAGE_H
