#ifndef HFETCH_NUMPYSTORAGE_H
#define HFETCH_NUMPYSTORAGE_H

#include "StorageInterface.h"
#include "ArrayPartitioner.h"


#include <python2.7/Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#include "numpy/arrayobject.h"


/***
 * Responsible to store a numpy to the keyspace.table_numpies, associating an attribute_name and a storage_id(uuid)
 */

class NumpyStorage {

public:

    NumpyStorage(std::shared_ptr<StorageInterface> storage, ArrayPartitioner &algorithm);

    ArrayMetadata* store(std::string table, std::string keyspace, std::string attr_name, const CassUuid &storage_id, PyArrayObject* numpy) const;

    PyObject* read(std::string table, TupleRow* keys, ArrayMetadata &np);

private:

    ArrayMetadata* get_np_metadata(PyArrayObject* numpy) const;

    std::shared_ptr<StorageInterface> storage;

    ArrayPartitioner partitioner;

};


#endif //HFETCH_NUMPYSTORAGE_H
