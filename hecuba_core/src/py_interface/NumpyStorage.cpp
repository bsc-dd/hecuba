#include "NumpyStorage.h"
#include "NumpyStorage.h"
#include <iostream>


NumpyStorage::NumpyStorage(const char *table, const char *keyspace, CassSession *session,
                           std::map<std::string, std::string> &config) :
        ArrayDataStore(table, keyspace, session, config) {


}


NumpyStorage::~NumpyStorage() {

};

std::list<std::vector<uint32_t> > NumpyStorage::generate_coords(PyObject *coord) const {
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

void NumpyStorage::store_numpy(const uint64_t *storage_id, ArrayMetadata &np_metas, PyArrayObject *numpy, PyObject *coord) const {
    void *data = PyArray_DATA(numpy);
    if (coord != Py_None) {
        std::list<std::vector<uint32_t> > crd = generate_coords(coord);
        this->store_numpy_into_cas_by_coords(storage_id, np_metas, data, crd);
    } else this->store_numpy_into_cas(storage_id, np_metas, data);
}

void NumpyStorage::load_numpy(const uint64_t *storage_id, ArrayMetadata &np_metas, PyArrayObject *save, PyObject *coord) {
    void *data = PyArray_DATA(save);
    if (coord != Py_None) {
        std::list<std::vector<uint32_t> > crd = generate_coords(coord);
        this->read_numpy_from_cas_by_coords(storage_id, np_metas, crd, data);
    } else this->read_numpy_from_cas(storage_id, np_metas, data);
}

PyObject *NumpyStorage::reserve_numpy_space(const uint64_t *storage_id, ArrayMetadata &np_metas) {
    npy_intp *dims = new npy_intp[np_metas.dims.size()];
    for (uint32_t i = 0; i < np_metas.dims.size(); ++i) {
        dims[i] = np_metas.dims[i];
    }

    int32_t type = PyArray_TypestrConvert(np_metas.elem_size, np_metas.typekind);
    if (type == NPY_NOTYPE)
        throw ModuleException("Can't identify Numpy dtype from typekind");

    PyObject *resulting_array;
    try {
	//yolandab
        int fortran_layout=0;
        if (np_metas.flags&NPY_ARRAY_F_CONTIGUOUS){
		fortran_layout=1;
	}else{
		fortran_layout=0;
        }
        resulting_array = PyArray_ZEROS((int32_t) np_metas.dims.size(), dims, type, fortran_layout);
        // it was : resulting_array = PyArray_ZEROS((int32_t) np_metas.dims.size(), dims, type, 0);
        PyArrayObject *converted_array;
        PyArray_OutputConverter(resulting_array, &converted_array);
        PyArray_ENABLEFLAGS(converted_array, NPY_ARRAY_OWNDATA);
    }
    catch (std::exception &e) {
        if (PyErr_Occurred()) PyErr_Print();
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    delete[] dims;
    return resulting_array;
}

PyObject *NumpyStorage::get_row_elements(const uint64_t *storage_id, ArrayMetadata &np_metas) {
    uint32_t ndims = (uint32_t) np_metas.dims.size();
    uint64_t block_size = BLOCK_SIZE - (BLOCK_SIZE % np_metas.elem_size);
    uint32_t row_elements = (uint32_t) std::floor(pow(block_size / np_metas.elem_size, (1.0 / ndims)));
    return Py_BuildValue("i", row_elements);
}
