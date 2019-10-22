#include "NumpyStorage.h"


NumpyStorage::NumpyStorage(const char *table, const char *keyspace, CassSession *session,
                           std::map<std::string, std::string> &config) :
        ArrayDataStore(table, keyspace, session, config) {


}


NumpyStorage::~NumpyStorage() {

};


void NumpyStorage::store_numpy(const uint64_t *storage_id, PyArrayObject *numpy) const {

    ArrayMetadata *np_metas = this->get_np_metadata(numpy);
    np_metas->partition_type = ZORDER_ALGORITHM;

    void *data = PyArray_BYTES(numpy);
    this->store(storage_id, np_metas, data);
    this->update_metadata(storage_id, np_metas);
    delete (np_metas);
}


/***
 * Reads a numpy ndarray by fetching the clusters independently
 * @param storage_id of the array to retrieve
 * @return Numpy ndarray as a Python object
 */
PyObject *NumpyStorage::read_numpy(const uint64_t *storage_id) {
    ArrayMetadata *np_metas = this->read_metadata(storage_id);
    void *data = this->read(storage_id, np_metas);


    npy_intp *dims = new npy_intp[np_metas->dims.size()];
    for (uint32_t i = 0; i < np_metas->dims.size(); ++i) {
        dims[i] = np_metas->dims[i];
    }

    PyObject *resulting_array;
    try {
        resulting_array = PyArray_SimpleNewFromData((int32_t) np_metas->dims.size(), dims, np_metas->inner_type, data);

        PyArrayObject *converted_array;
        PyArray_OutputConverter(resulting_array, &converted_array);
        PyArray_ENABLEFLAGS(converted_array, NPY_ARRAY_OWNDATA);
    }
    catch (std::exception e) {
        if (PyErr_Occurred()) PyErr_Print();
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    delete (np_metas);

    return resulting_array;
}

/***
 * Extract the metadatas of the given numpy ndarray and return a new ArrayMetadata with its representation
 * @param numpy Ndarray to extract the metadata
 * @return ArrayMetadata defining the information to reconstruct a numpy ndarray
 */
ArrayMetadata *NumpyStorage::get_np_metadata(PyArrayObject *numpy) const {
    int64_t ndims = PyArray_NDIM(numpy);
    npy_intp *shape = PyArray_SHAPE(numpy);

    ArrayMetadata *shape_and_type = new ArrayMetadata();
    shape_and_type->inner_type = PyArray_TYPE(numpy);

    //TODO implement as a union
    if (shape_and_type->inner_type == NPY_INT8) shape_and_type->elem_size = sizeof(int8_t);
    else if (shape_and_type->inner_type == NPY_UINT8) shape_and_type->elem_size = sizeof(uint8_t);
    else if (shape_and_type->inner_type == NPY_INT16) shape_and_type->elem_size = sizeof(int16_t);
    else if (shape_and_type->inner_type == NPY_UINT16) shape_and_type->elem_size = sizeof(uint16_t);
    else if (shape_and_type->inner_type == NPY_INT32) shape_and_type->elem_size = sizeof(int32_t);
    else if (shape_and_type->inner_type == NPY_UINT32) shape_and_type->elem_size = sizeof(uint32_t);
    else if (shape_and_type->inner_type == NPY_INT64) shape_and_type->elem_size = sizeof(int64_t);
    else if (shape_and_type->inner_type == NPY_LONGLONG) shape_and_type->elem_size = sizeof(int64_t);
    else if (shape_and_type->inner_type == NPY_UINT64) shape_and_type->elem_size = sizeof(uint64_t);
    else if (shape_and_type->inner_type == NPY_ULONGLONG) shape_and_type->elem_size = sizeof(uint64_t);
    else if (shape_and_type->inner_type == NPY_DOUBLE) shape_and_type->elem_size = sizeof(npy_double);
    else if (shape_and_type->inner_type == NPY_FLOAT16) shape_and_type->elem_size = sizeof(npy_float16);
    else if (shape_and_type->inner_type == NPY_FLOAT32) shape_and_type->elem_size = sizeof(npy_float32);
    else if (shape_and_type->inner_type == NPY_FLOAT64) shape_and_type->elem_size = sizeof(npy_float64);
    else if (shape_and_type->inner_type == NPY_FLOAT128) shape_and_type->elem_size = sizeof(npy_float128);
    else if (shape_and_type->inner_type == NPY_BOOL) shape_and_type->elem_size = sizeof(bool);
    else if (shape_and_type->inner_type == NPY_BYTE) shape_and_type->elem_size = sizeof(char);
    else if (shape_and_type->inner_type == NPY_LONG) shape_and_type->elem_size = sizeof(long);
    else if (shape_and_type->inner_type == NPY_LONGLONG) shape_and_type->elem_size = sizeof(long long);
    else if (shape_and_type->inner_type == NPY_SHORT) shape_and_type->elem_size = sizeof(short);
    else throw ModuleException("Numpy data type still not supported");

    // Copy elements per dimension
    shape_and_type->dims = std::vector<uint32_t>((uint64_t) ndims);//PyArray_SHAPE()
    for (int32_t dim = 0; dim < ndims; ++dim) {
        shape_and_type->dims[dim] = (uint32_t) shape[dim];
    }
    return shape_and_type;
}
