#ifndef HFETCH_NUMPYSTORAGE_H
#define HFETCH_NUMPYSTORAGE_H

#include "../ArrayDataStore.h"

#include <Python.h>
#include <climits>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include "numpy/arrayobject.h"

/***
 * Responsible to store a numpy to the keyspace.table_numpies, associating an attribute_name and a storage_id(uuid)
 */

class NumpyStorage : public ArrayDataStore {

public:

    NumpyStorage(CacheTable *cache, CacheTable *read_cache,
                 std::map<std::string, std::string> &config);

    ~NumpyStorage();

    void store_numpy(const uint64_t *storage_id, PyArrayObject *numpy) const;

    PyObject *read_numpy(const uint64_t *storage_id);
    PyObject *reserve_numpy_space(const uint64_t *storage_id);
    PyObject *coord_list_to_numpy(const uint64_t *storage_id, PyObject * coord, char* save);

private:

    ArrayMetadata *get_np_metadata(PyArrayObject *numpy) const;



};


#endif //HFETCH_NUMPYSTORAGE_H
