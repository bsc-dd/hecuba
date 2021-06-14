#ifndef HFETCH_NUMPYSTORAGE_H
#define HFETCH_NUMPYSTORAGE_H

#include "../ArrayDataStore.h"
#include "../MetaManager.h"
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

    NumpyStorage(const char *table, const char *keyspace, std::shared_ptr<StorageInterface> storage,
                 std::map<std::string, std::string> &config);

    ~NumpyStorage();

    std::list<std::vector<uint32_t> > generate_coords(PyObject *coord) const;

    PyObject *reserve_numpy_space(const uint64_t *storage_id, ArrayMetadata &np_metas);

    PyObject *get_row_elements(const uint64_t *storage_id, ArrayMetadata &np_metas);

    void store_numpy(const uint64_t *storage_id, ArrayMetadata &, PyArrayObject *numpy, PyObject *coord) const;

    void load_numpy(const uint64_t *storage_id, ArrayMetadata &np_metas, PyArrayObject *save, PyObject *coord);

    void load_numpy_arrow(const uint64_t *storage_id, ArrayMetadata &np_metas, PyArrayObject *save, PyObject *cols);
    std::vector<uint64_t> get_cols(PyObject *coord) const;

private:

    MetaManager *MM;

    //ArrayMetadata *get_np_metadata(PyArrayObject *numpy) const;

};


#endif //HFETCH_NUMPYSTORAGE_H
