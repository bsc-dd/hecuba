#include "NumpyStorage.h"
#include "NumpyStorage.h"
#include <iostream>


#define BLOCK_MODE 1
#define COLUMN_MODE 2

NumpyStorage::NumpyStorage(const char *table, const char *keyspace, std::shared_ptr<StorageInterface> storage,
                 std::map<std::string, std::string> &config) :
        ArrayDataStore(table, keyspace, storage, config) {

    //lgarrobe preparant per metadates
    std::vector<std::map<std::string, std::string> > keys_names = {{{"name", "storage_id"}}};

    std::vector<std::map<std::string, std::string> > columns_names = {{{"name", "name"}},{{"name", "numpy_meta"}}};

    CassSession* session = storage->get_session();
    TableMetadata *table_meta = new TableMetadata("istorage", "hecuba", keys_names, columns_names, session);
    this ->MM = new MetaManager(table_meta, session, config);
}

NumpyStorage::~NumpyStorage() {

};

std::list<std::vector<uint32_t> > NumpyStorage::generate_coords(PyObject *coord) const {
    std::vector<uint32_t> crd_inner = {};
    std::list<std::vector<uint32_t> > crd = {};
    if (PyList_Check(coord)) {
        if (PyList_Size(coord) == 0) return crd; //No elements in list...
        crd_inner.resize((PyTuple_Size(PyList_GetItem(coord, 0))));
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

// Transform a Python list of columns to a vector of column IDs (uints)
std::vector<uint64_t> NumpyStorage::get_cols(PyObject *coord) const {
	std::vector<uint64_t> c = {};
	if (PyList_Check(coord)) {
		PyObject *value = nullptr;
		for (Py_ssize_t i = 0; i < PyList_Size(coord); i++) {
			value = PyList_GetItem(coord, i);
			c.push_back(PyLong_AsLong(value));
		}
	}
	return c;
}

void NumpyStorage::store_numpy(const uint64_t *storage_id, ArrayMetadata &np_metas, PyArrayObject *numpy, PyObject *coord, int py_order) const {
    void *data = PyArray_DATA(numpy);
	if (py_order == BLOCK_MODE) {
		if (coord != Py_None) {
			std::list<std::vector<uint32_t> > crd = generate_coords(coord);
			this->store_numpy_into_cas_by_coords(storage_id, np_metas, data, crd);
		} else {
			this->store_numpy_into_cas(storage_id, np_metas, data);
		}
	} else { // COLUMN_MODE
		if (coord != Py_None) {
			// FIXME NOT WORKING
			throw ModuleException("Storing a column range is NOT IMPLEMENTED");
			//this->store_numpy_into_cas_by_cols_as_arrow(storage_id, np_metas, data, get_cols(coord));
		}else {
			this->store_numpy_into_cas_as_arrow(storage_id, np_metas, data);
		}
	}
}

void NumpyStorage::load_numpy(const uint64_t *storage_id, ArrayMetadata &np_metas, PyArrayObject *save, PyObject *coord, int py_order) {
	void *data = PyArray_DATA(save);
	if (py_order == BLOCK_MODE) {
		if (coord != Py_None) {
			std::list<std::vector<uint32_t> > crd = generate_coords(coord);
			this->read_numpy_from_cas_by_coords(storage_id, np_metas, crd, data);
		} else this->read_numpy_from_cas(storage_id, np_metas, data);

	} else { // COLUMN_MODE
		std::vector<uint64_t> c = {};
		if (coord != Py_None) {
			c = get_cols( coord );
		} else {
			// None == ALL => Build a list with ALL COLUMNS #FIXME Avoid creating this list!
			for(uint64_t i = 0 ; i < np_metas.dims[np_metas.dims.size()-2]; i++) { // Traverse all the columns --> dims[-2]
				c.push_back(i);
			}
		}
		this->read_numpy_from_cas_arrow(storage_id, np_metas, c, data);
	}
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

/***
 * Extract the metadatas of the given numpy ndarray and return a new ArrayMetadata with its representation
 * @param numpy Ndarray to extract the metadata
 * @return ArrayMetadata defining the information to reconstruct a numpy ndarray
 */
/*
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
*/
