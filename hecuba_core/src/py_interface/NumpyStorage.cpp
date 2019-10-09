#include "NumpyStorage.h"


NumpyStorage::NumpyStorage(CacheTable *cache, CacheTable *read_cache,
                           std::map<std::string, std::string> &config) : ArrayDataStore(cache, read_cache, config) {


}


NumpyStorage::~NumpyStorage() {

};

std::list<std::vector<uint32_t> > NumpyStorage::generate_coords(PyObject * coord) const {
    std::vector<uint32_t> crd_inner = {};
    std::list<std::vector<uint32_t> > crd = {};
    crd_inner.resize((PyTuple_Size(PyList_GetItem(coord, 0))));
    if (PyList_Check(coord)) {
        PyObject *value = nullptr;
        for (Py_ssize_t i = 0; i < PyList_Size(coord); i++) {
            value = PyList_GetItem(coord, i);
            for (Py_ssize_t j = 0; j < PyTuple_Size(value); j++) {
                crd_inner[j] = (PyLong_AsLong(PyTuple_GetItem(value, j)));
            }
            crd.push_back(crd_inner);
        }
    }
    return crd;
}

void *NumpyStorage::store_numpy_by_coord(const uint64_t *storage_id, PyArrayObject *numpy, PyObject *coord) const {
    ArrayMetadata *np_metas = this->get_np_metadata(numpy);
    np_metas->partition_type = ZORDER_ALGORITHM;
    void *data = PyArray_DATA(numpy);
    std::list<std::vector<uint32_t> > crd = {};
    if (coord != Py_None) {
        crd = generate_coords(coord);
        this->store_numpy_into_cas_by_coords(storage_id, np_metas, data, crd);
    }
    else this->store_numpy_into_cas(storage_id, np_metas, data);
    this->update_metadata(storage_id, np_metas);
    delete (np_metas);
}

void *NumpyStorage::load_numpy_from_coord(const uint64_t *storage_id, PyObject *coord, PyArrayObject *save) {
    ArrayMetadata *np_metas = this->get_np_metadata(save);
    np_metas->partition_type = ZORDER_ALGORITHM;
    void *data = PyArray_DATA(save);
    std::list<std::vector<uint32_t> > crd = {};
    if (coord != Py_None) {
        crd = generate_coords(coord);
        this->read_numpy_from_cas_by_coords(storage_id, np_metas, crd, data);
    }
    else this->read_numpy_from_cas(storage_id, np_metas, crd, data);
    this->update_metadata(storage_id, np_metas);
    delete (np_metas);
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

PyObject *NumpyStorage::get_row_elements(const uint64_t *storage_id) {
    ArrayMetadata *np_metas = this->read_metadata(storage_id);
    uint32_t ndims = (uint32_t) np_metas->dims.size();
    uint64_t block_size = BLOCK_SIZE - (BLOCK_SIZE % np_metas->elem_size);
    uint32_t row_elements = (uint32_t) std::floor(pow(block_size / np_metas->elem_size, (1.0 / ndims)));
    return Py_BuildValue("i",row_elements);
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
