#include "NumpyStorage.h"


NumpyStorage::NumpyStorage(const char *table, const char *keyspace, CassSession *session,
                           std::map<std::string, std::string> &config) :
        ArrayDataStore(table, keyspace, session, config) {


}


NumpyStorage::~NumpyStorage() {

};


void NumpyStorage::store_numpy(const uint64_t *storage_id, PyArrayObject *numpy, ArrayMetadata &np_metas) const {
    void *data = PyArray_BYTES(numpy);
    this->store(storage_id, np_metas, data);
}


/***
 * Reads a numpy ndarray by fetching the clusters independently
 * @param storage_id of the array to retrieve
 * @param np_metas array description
 * @return Numpy ndarray as a Python object
 */
PyObject *NumpyStorage::read_numpy(const uint64_t *storage_id, ArrayMetadata &np_metas) {
    void *data = this->read(storage_id, np_metas);

    npy_intp *dims = new npy_intp[np_metas.dims.size()];
    for (uint32_t i = 0; i < np_metas.dims.size(); ++i) {
        dims[i] = np_metas.dims[i];
    }

    int32_t type = PyArray_TypestrConvert(np_metas.elem_size, np_metas.typekind);
    if (type == NPY_NOTYPE)
        throw ModuleException("Can't identify Numpy dtype from typekind");

    PyObject *resulting_array;
    try {
        resulting_array = PyArray_SimpleNewFromData((int32_t) np_metas.dims.size(), dims, type, data);

        PyArrayObject *converted_array;
        PyArray_OutputConverter(resulting_array, &converted_array);
        PyArray_ENABLEFLAGS(converted_array, NPY_ARRAY_OWNDATA);
    }
    catch (std::exception &e) {
        if (PyErr_Occurred()) PyErr_Print();
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    return resulting_array;
}

