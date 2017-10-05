#ifndef HFETCH_NUMPYSTORAGE_H
#define HFETCH_NUMPYSTORAGE_H

#include "StorageInterface.h"
#include "SpaceFillingCurve.h"


#include <python2.7/Python.h>
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

    NumpyStorage(std::string table, std::string keyspace, std::shared_ptr<StorageInterface> storage);

    ~NumpyStorage();

    const ArrayMetadata *store(const CassUuid &storage_id, PyArrayObject *numpy) const;

    PyObject *read(std::string table, std::string keyspace, const CassUuid &storage_id, const ArrayMetadata *arr_meta);

private:

    ArrayMetadata *get_np_metadata(PyArrayObject *numpy) const;

    std::shared_ptr<StorageInterface> storage;
    Writer *writer;

    SpaceFillingCurve partitioner;

};


#endif //HFETCH_NUMPYSTORAGE_H
