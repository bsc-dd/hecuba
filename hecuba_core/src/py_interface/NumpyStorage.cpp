#include "NumpyStorage.h"


NumpyStorage::NumpyStorage(CacheTable *cache, CacheTable *read_cache,
                           std::map<std::string, std::string> &config) : ArrayDataStore(cache, read_cache, config) {


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

PyObject *NumpyStorage::coord_list_to_numpy(const uint64_t *storage_id, PyObject *coord, PyArrayObject *save) {
    ArrayMetadata *np_metas = this->read_metadata(storage_id);
    void *data = PyArray_DATA(save);
    npy_intp *dims = new npy_intp[np_metas->dims.size()];

    std::vector<uint32_t> crd_inner;
    std::vector<std::vector<uint32_t> > crd;


    if (coord != Py_None) {
        crd.resize(PyList_Size(coord), std::vector<uint32_t>(PyList_Size(PyList_GetItem(coord, 0))));
        crd_inner.resize((PyList_Size(PyList_GetItem(coord, 0))));
        uint32_t ndims = (uint32_t) np_metas->dims.size();
        uint64_t block_size = BLOCK_SIZE - (BLOCK_SIZE % np_metas->elem_size);
        uint32_t row_elements = (uint32_t) std::floor(pow(block_size / np_metas->elem_size, (1.0 / ndims)));
        if (PyList_Check(coord)) {
            for (Py_ssize_t i = 0; i < PyList_Size(coord); i++) {
                PyObject *value = PyList_GetItem(coord, i);
                for (Py_ssize_t j = 0; j < PyList_Size(value); j++) {
                    crd_inner[j] = PyLong_AsLong(PyList_GetItem(value, j)) / row_elements;
                }
                crd[i] = crd_inner;
            }
        }
    } else crd = {};
    void *numpy_data = this->read_n_coord(storage_id, np_metas, crd, data);
    for (uint32_t i = 0; i < np_metas->dims.size(); ++i) {
        dims[i] = np_metas->dims[i];
    }
    PyObject *result;
    try {
        result = PyArray_SimpleNewFromData((int32_t) np_metas->dims.size(), dims, np_metas->inner_type, numpy_data);
        PyArrayObject *converted_array;
        PyArray_OutputConverter(result, &converted_array);
        PyArray_ENABLEFLAGS(converted_array, NPY_ARRAY_OWNDATA);
    }
    catch (std::exception e) {
        if (PyErr_Occurred()) PyErr_Print();
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    delete (np_metas);
    return result;
}

PyObject *NumpyStorage::reserve_numpy_space(const uint64_t *storage_id) {
    ArrayMetadata *np_metas = this->read_metadata(storage_id);

    npy_intp *dims = new npy_intp[np_metas->dims.size()];
    for (uint32_t i = 0; i < np_metas->dims.size(); ++i) {
        dims[i] = np_metas->dims[i];
    }
    PyObject *resulting_array;
    try {
        resulting_array = PyArray_ZEROS((int32_t) np_metas->dims.size(), dims, np_metas->inner_type, 0);
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
 * Reads a numpy ndarray by fetching the clusters indipendently
 * @param storage_id of the array to retrieve
 * @return Numpy ndarray as a Python object
 */
PyObject *NumpyStorage::read_numpy(const uint64_t *storage_id) {
    ArrayMetadata *np_metas = this->read_metadata(storage_id);
    void *data = this->read_n_coord(storage_id, np_metas, {}, nullptr);

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
