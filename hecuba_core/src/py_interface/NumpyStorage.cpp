#include "NumpyStorage.h"


NumpyStorage::NumpyStorage(const char *table, const char *keyspace, CassSession *session,
                           std::map<std::string, std::string> &config) :
        ArrayDataStore(table, keyspace, session, config) {


}


NumpyStorage::~NumpyStorage() {

};


void NumpyStorage::store_numpy(const uint64_t *storage_id, PyArrayObject *numpy, ArrayMetadata *np_metas) const {
    void *data = PyArray_BYTES(numpy);
    this->store(storage_id, np_metas, data);
}


/***
 * Reads a numpy ndarray by fetching the clusters independently
 * @param storage_id of the array to retrieve
 * @param np_metas array description
 * @return Numpy ndarray as a Python object
 */
PyObject *NumpyStorage::read_numpy(const uint64_t *storage_id, ArrayMetadata *np_metas) {
    void *data = this->read(storage_id, np_metas);

    npy_intp *dims = new npy_intp[np_metas->dims.size()];
    for (uint32_t i = 0; i < np_metas->dims.size(); ++i) {
        dims[i] = np_metas->dims[i];
    }

    int32_t type = PyArray_TypestrConvert(np_metas->elem_size, np_metas->typekind[0]);
    if (type == NPY_NOTYPE)
        throw ModuleException("Can't identify Numpy dtype from typekind");

    PyObject *resulting_array;
    try {
        resulting_array = PyArray_SimpleNewFromData((int32_t) np_metas->dims.size(), dims, type, data);

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


ArrayMetadata *NumpyStorage::make_metadata(PyObject *py_np_metas) const {

    const char *expected = "np_meta"; //hecuba.hnumpy.StorageNumpy.
    if (strlen(py_np_metas->ob_type->tp_name) != strlen(expected) ||
        memcmp(py_np_metas->ob_type->tp_name, expected, strlen(expected)) != 0) {
        PyErr_SetString(PyExc_ValueError, "Numpy metadata class does not match the expected");
        std::cerr << py_np_metas->ob_type->tp_name << std::endl;
        return NULL;
    }

    ArrayMetadata *np_meta = new ArrayMetadata();

    // elem_size (itemsize)
    PyObject *attr = PyObject_GetAttrString(py_np_metas, "elem_size");
    if (attr == Py_None) return nullptr;
    if (!PyLong_Check(attr) || !PyArg_Parse(attr, Py_INT, &np_meta->elem_size))
        throw ModuleException("Numpy elem_size must be an int");
    Py_DECREF(attr);



    // Dims and ndims
    attr = PyObject_GetAttrString(py_np_metas, "dims");
    if (attr == Py_None || !PyList_Check(attr)) return nullptr;
    int32_t ndims = PyList_Size(attr);
    np_meta->dims.resize(ndims);
    np_meta->strides.resize(ndims);


    for (uint32_t dim_i = 0; dim_i < ndims; ++dim_i) {
        PyObject *elem_dim = PyList_GetItem(attr, dim_i);
        if (elem_dim == Py_None) return nullptr;
        if (!PyLong_Check(elem_dim) || !PyArg_Parse(elem_dim, Py_INT, &np_meta->dims[dim_i]))
            throw ModuleException("Numpy dims must be a list of ints");
    }
    Py_DECREF(attr);


    // Strides
    attr = PyObject_GetAttrString(py_np_metas, "strides");
    if (attr == Py_None || !PyList_Check(attr)) return nullptr;
    else {
        if (PyList_Size(attr) != ndims) throw ModuleException("Numpy strides must be a list of ints");
        for (uint32_t dim_i = 0; dim_i < ndims; ++dim_i) {
            PyObject *elem_dim = PyList_GetItem(attr, dim_i);
            if (elem_dim == Py_None) return nullptr;
            if (!PyLong_Check(elem_dim) || !PyArg_Parse(elem_dim, Py_INT, &np_meta->strides[dim_i]))
                throw ModuleException("Numpy strides must be a list of ints");
        }
    }
    Py_DECREF(attr);


    // Typekind (gentype) != typecode
    attr = PyObject_GetAttrString(py_np_metas, "typekind");
    if (attr == Py_None) return nullptr;

    if (PyUnicode_Check(attr)) {
        Py_ssize_t l_size;
        const char *l_temp = PyUnicode_AsUTF8AndSize(attr, &l_size);
        if (!l_temp)
            throw ModuleException("Numpy typekind must be UTF8");
        np_meta->typekind = std::string(l_temp, l_size + 1);
    }
    Py_DECREF(attr);

    // Byteorder
    attr = PyObject_GetAttrString(py_np_metas, "byteorder");
    if (attr == Py_None) return nullptr;

    if (PyUnicode_Check(attr)) {
        Py_ssize_t l_size;
        const char *l_temp = PyUnicode_AsUTF8AndSize(attr, &l_size);
        if (!l_temp)
            throw ModuleException("Byteorder must be UTF8");
        np_meta->byteorder = std::string(l_temp, l_size + 1);
    }
    Py_DECREF(attr);


    // Flags
    attr = PyObject_GetAttrString(py_np_metas, "flags");
    if (attr == Py_None) np_meta->flags = 0;
    else if (!PyLong_Check(attr) || !PyArg_Parse(attr, Py_INT, &np_meta->flags))
        throw ModuleException("Numpy flgs must be an int");
    Py_DECREF(attr);

    // Partition algorithm
    attr = PyObject_GetAttrString(py_np_metas, "partition_type");
    if (attr == Py_None) np_meta->partition_type = NO_PARTITIONS;
    else if (!PyLong_Check(attr) || !PyArg_Parse(attr, Py_SHORT_INT, &np_meta->partition_type))
        throw ModuleException("Numpy partition type must be an int");
    Py_DECREF(attr);

    return np_meta;
}
