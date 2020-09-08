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

    NumpyStorage(const char *table, const char *keyspace, CassSession *session,
                 std::map<std::string, std::string> &config);

    ~NumpyStorage();

    std::list<std::vector<uint32_t> > generate_coords(PyObject *coord) const;

    PyObject *reserve_numpy_space(const uint64_t *storage_id, ArrayMetadata &np_metas);

    PyObject *get_row_elements(const uint64_t *storage_id, ArrayMetadata &np_metas);

    void store_numpy(const uint64_t *storage_id, ArrayMetadata &, PyArrayObject *numpy, PyObject *coord) const;

    void load_numpy(const uint64_t *storage_id, ArrayMetadata &np_metas, PyArrayObject *save, PyObject *coord);

private:

    MetaManager *MM;

    //ArrayMetadata *get_np_metadata(PyArrayObject *numpy) const;

};


#endif //HFETCH_NUMPYSTORAGE_H
