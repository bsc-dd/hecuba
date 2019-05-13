#ifndef HFETCH_NUMPYSTORAGE_H
#define HFETCH_NUMPYSTORAGE_H

#include "../StorageInterface.h"
#include "../SpaceFillingCurve.h"


#include <Python.h>
#include <climits>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include "numpy/arrayobject.h"

/***
 * Responsible to store a numpy to the keyspace.table_numpies, associating an attribute_name and a storage_id(uuid)
 */

class NumpyStorage {

public:

    NumpyStorage(const TableMetadata *table_meta, std::shared_ptr<StorageInterface> storage,
                 std::map<std::string, std::string> &config);

    ~NumpyStorage();

    void store(const uint64_t *storage_id, PyArrayObject *numpy) const;

    PyObject *read(const uint64_t *storage_id);

    PyObject *read_by_tokens(const uint64_t *storage_id, const std::vector<std::pair<int64_t, int64_t>> &tokens);


private:

    const ArrayMetadata *read_array_meta(const uint64_t *storage_id, CacheTable *cache) const;

    void store_array_meta(const uint64_t *storage_id, ArrayMetadata *np_metas) const;

    void store_entire_array(const uint64_t *storage_id, ArrayMetadata *np_metas, PyArrayObject *numpy) const;

    ArrayMetadata *get_np_metadata(PyArrayObject *numpy) const;

    std::shared_ptr<StorageInterface> storage;
    Writer *writer;

    SpaceFillingCurve partitioner;

};


#endif //HFETCH_NUMPYSTORAGE_H
