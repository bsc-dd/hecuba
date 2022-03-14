#include "HCache.h"


/** MODULE METHODS **/

static PyObject *connectCassandra(PyObject *self, PyObject *args) {
    int nodePort;
    std::string contact_points = "";
    PyObject *py_contact_points;

    if (!PyArg_ParseTuple(args, "Oi", &py_contact_points, &nodePort)) {
        return NULL;
    }

    uint16_t contact_p_len = (uint16_t) PyList_Size(py_contact_points);

    for (uint16_t i = 0; i < contact_p_len; ++i) {
        char *str_temp;
        if (!PyArg_Parse(PyList_GetItem(py_contact_points, i), "s", &str_temp)) {
            PyErr_SetString(PyExc_TypeError, "Invalid contact point for Cassandra, not a string");
            return NULL;
        };
        if (!strlen(str_temp)) {
            PyErr_SetString(PyExc_ValueError, "Empty string as a contact point is invalid");
            return NULL;
        }
        contact_points += std::string(str_temp) + ",";
    }

    try {
        storage = std::make_shared<StorageInterface>(nodePort, contact_points);
        //TODO storage = new StorageInterface(nodePort, contact_points);
    }
    catch (ModuleException &e) {
        PyErr_SetString(PyExc_OSError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *disconnectCassandra(PyObject *self) {
    if (storage != NULL)
        storage->disconnectCassandra();
    Py_RETURN_NONE;
}


/*** HCACHE DATA TYPE METHODS AND SETUP ***/

static PyObject *put_row(HCache *self, PyObject *args) {
    PyObject *py_keys, *py_values;
    if (!PyArg_ParseTuple(args, "OO", &py_keys, &py_values)) {
        return NULL;
    }
    for (uint16_t key_i = 0; key_i < PyList_Size(py_keys); ++key_i) {
        if (PyList_GetItem(py_keys, key_i) == Py_None) {
            std::string error_msg = "Keys can't be None, key_position: " + std::to_string(key_i);
            PyErr_SetString(PyExc_TypeError, error_msg.c_str());
            return NULL;
        }
    }
    TupleRow *k;
    try {
        k = self->keysParser->make_tuple(py_keys);
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        std::string error_msg = "Put_row, keys error: " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    }
    try {
        TupleRow *v = self->valuesParser->make_tuple(py_values);
        self->T->put_crow(k, v);
        delete (k);
        delete (v);
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        std::string err_msg = "Put row " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, err_msg.c_str());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *get_row(HCache *self, PyObject *args) {
    PyObject *py_keys, *py_row;
    if (!PyArg_ParseTuple(args, "O", &py_keys)) {
        return NULL;
    }

    for (uint16_t key_i = 0; key_i < PyList_Size(py_keys); ++key_i) {
        if (PyList_GetItem(py_keys, key_i) == Py_None) {
            std::string error_msg = "Keys can't be None, key_position: " + std::to_string(key_i);
            PyErr_SetString(PyExc_TypeError, error_msg.c_str());
            return NULL;
        }
    }

    TupleRow *k = NULL;

    try {
        k = self->keysParser->make_tuple(py_keys);
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        std::string error_msg = "Get row, keys error: " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    }
    std::vector<const TupleRow *> v;
    try {
        v = self->T->get_crow(k);
        delete (k);
    }
    catch (std::exception &e) {
        std::string error_msg = "Get row error: " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    if (v.empty()) {
        PyErr_SetString(PyExc_KeyError, "No values found for this key");
        return NULL;
    }

    try {
        if (self->T->get_metadata()->get_values()->empty()) {
            std::vector<const TupleRow *> empty_result(0);
        }
        py_row = self->valuesParser->make_pylist(v);
        for (uint32_t i = 0; i < v.size(); ++i) {
            delete (v[i]);
        }
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        std::string error_msg = "Get row, values error: " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    return py_row;
}


static PyObject *delete_row(HCache *self, PyObject *args) {
    PyObject *py_keys;
    if (!PyArg_ParseTuple(args, "O", &py_keys)) {
        return NULL;
    }
    for (uint16_t key_i = 0; key_i < PyList_Size(py_keys); ++key_i) {
        if (PyList_GetItem(py_keys, key_i) == Py_None) {
            std::string error_msg = "Keys can't be None, key_position: " + std::to_string(key_i);
            PyErr_SetString(PyExc_TypeError, error_msg.c_str());
            return NULL;
        }
    }
    TupleRow *k;
    try {
        k = self->keysParser->make_tuple(py_keys);
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        std::string error_msg = "Delete_row, keys error: " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    }
    try {
        self->T->delete_crow(k);
        delete (k);
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        std::string err_msg = "Delete row " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, err_msg.c_str());
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject *flush(HCache *self, PyObject *args) {
    try {
        self->T->flush_elements();
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        std::string err_msg = "Flushing elements failed with " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, err_msg.c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

static void hcache_dealloc(HCache *self) {
    delete (self->keysParser);
    delete (self->valuesParser);
    delete (self->T);
    Py_TYPE((PyObject *) self)->tp_free((PyObject *) self);
}


static PyObject *hcache_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    HCache *self;
    self = (HCache *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}


static int hcache_init(HCache *self, PyObject *args, PyObject *kwds) {
    const char *table, *keyspace;
    PyObject *py_tokens, *py_keys_names, *py_cols_names, *py_config, *py_storage_id;

    if (!PyArg_ParseTuple(args, "ssOOOOO", &keyspace, &table, &py_storage_id, &py_tokens,
                          &py_keys_names, &py_cols_names, &py_config)) {
        return -1;
    };


    /** PARSE CONFIG **/

    std::map<std::string, std::string> config;

    if (PyDict_Check(py_config)) {
        PyObject *dict, *key, *value;;
        if (!PyArg_Parse(py_config, "O", &dict)) return -1;

        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)) {
            std::string conf_key(PyUnicode_AsUTF8(key));
            if (PyUnicode_Check(value)) {
                std::string conf_val(PyUnicode_AsUTF8(value));
                config[conf_key] = conf_val;
            }
            if (PyLong_Check(value)) {
                int32_t c_val = (int32_t) PyLong_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }

        }
    }

    /*** PARSE TABLE METADATA ***/

    uint16_t tokens_size = (uint16_t) PyList_Size(py_tokens);
    uint16_t keys_size = (uint16_t) PyList_Size(py_keys_names);
    uint16_t cols_size = (uint16_t) PyList_Size(py_cols_names);

    int64_t t_a, t_b;
    self->token_ranges = std::vector<std::pair<int64_t, int64_t >>(tokens_size);
    for (uint16_t i = 0; i < tokens_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_tokens, i);
        if (!PyArg_ParseTuple(obj_to_convert, "LL", &t_a, &t_b)) return -1;
        self->token_ranges[i] = std::make_pair(t_a, t_b);
    }


    std::vector<std::map<std::string, std::string>> keys_names(keys_size);

    for (uint16_t i = 0; i < keys_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_keys_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        }
        keys_names[i] = {{"name", std::string(str_temp)}};
    }

    std::vector<std::map<std::string, std::string>> columns_names(cols_size);
    for (uint16_t i = 0; i < cols_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_cols_names, i);

        if (PyUnicode_Check(obj_to_convert)) {
            char *str_temp;
            if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
                return -1;
            };
            columns_names[i] = {{"name", std::string(str_temp)}};
        } else if (PyDict_Check(obj_to_convert)) {
            PyObject *dict;
            if (!PyArg_Parse(obj_to_convert, "O", &dict)) {
                return -1;
            };

            PyObject *py_name = PyDict_GetItemString(dict, "name");
            columns_names[i]["name"] = PyUnicode_AsUTF8(py_name);
        } else {
            PyErr_SetString(PyExc_TypeError, "Can't parse column names, expected String, Dict or Unicode");
            return -1;
        }
    }


    try {
        self->T = storage->make_cache(table, keyspace, keys_names, columns_names, config);
        self->keysParser = new PythonParser(storage, self->T->get_metadata()->get_keys());
        self->valuesParser = new PythonParser(storage, self->T->get_metadata()->get_values());
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    return 0;
}


static PyMethodDef hcache_type_methods[] = {
        {"get_row",    (PyCFunction) get_row,            METH_VARARGS, NULL},
        {"put_row",    (PyCFunction) put_row,            METH_VARARGS, NULL},
        {"delete_row", (PyCFunction) delete_row,         METH_VARARGS, NULL},
        {"flush",      (PyCFunction) flush,              METH_VARARGS, NULL},
        {"iterkeys",   (PyCFunction) create_iter_keys,   METH_VARARGS, NULL},
        {"itervalues", (PyCFunction) create_iter_values, METH_VARARGS, NULL},
        {"iteritems",  (PyCFunction) create_iter_items,  METH_VARARGS, NULL},
        {NULL,         NULL,                             0,            NULL}
};


static PyTypeObject hfetch_HCacheType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "hfetch.Hcache",             /* tp_name */
        sizeof(HCache), /* tp_basicsize */
        0,                         /*tp_itemsize*/
        (destructor) hcache_dealloc, /*tp_dealloc*/
        0,                         /*tp_print*/
        0,                         /*tp_getattr*/
        0,                         /*tp_setattr*/
        0,                         /*tp_compare*/
        0,                         /*tp_repr*/
        0,                         /*tp_as_number*/
        0,                         /*tp_as_sequence*/
        0,                         /*tp_as_mapping*/
        0,                         /*tp_hash */
        0,                         /*tp_call*/
        0,                         /*tp_str*/
        0,                         /*tp_getattro*/
        0,                         /*tp_setattro*/
        0,                         /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
        "Cassandra rows",           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,                       /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        hcache_type_methods,             /* tp_methods */
        0,             /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc) hcache_init,      /* tp_init */
        0,                         /* tp_alloc */
        hcache_new,                 /* tp_new */
};


/*** NUMPY DATA STORE METHODS AND SETUP ***/

/***
 * Receives an UUID as a ByteArray or string and returns the representation in a c-like buffer
 * @param py_storage_id ByteArray or Python String representation of an UUID
 * @return C-like UUID
 */
static uint64_t *parse_uuid(PyObject *py_storage_id) {
    uint64_t *uuid;
    if (!PyByteArray_Check(py_storage_id)) {
        //Object is UUID python class
        uint32_t len = sizeof(uint64_t) * 2;
        uuid = (uint64_t *) malloc(len);

        PyObject *bytes = PyObject_GetAttrString(py_storage_id, "time_low"); //32b
        if (!bytes) throw TypeErrorException("Error parsing python UUID");

        bytes = PyObject_GetAttrString(py_storage_id, "bytes"); //64b
        if (!bytes)
            throw TypeErrorException("python UUID bytes not found");
        char *uuidnum = PyBytes_AsString(bytes);
        if (!uuidnum)
            throw TypeErrorException("unable to get python UUID bytes");
        memcpy(uuid, uuidnum, 16); // Keep the UUID as is (RFC4122)

    } else {
        uint32_t len = sizeof(uint64_t) * 2;
        uint32_t len_found = (uint32_t) PyByteArray_Size(py_storage_id);
        if (len_found != len) {
            std::string error_msg = "UUID received has size " + std::to_string(len_found) +
                                    ", expected was: " + std::to_string(len);
            PyErr_SetString(PyExc_ValueError, error_msg.c_str());
        }

        uuid = (uint64_t *) PyByteArray_AsString(py_storage_id);
    }
    return uuid;
}

static PyObject *get_elements_per_row(HNumpyStore *self, PyObject *args) {
    PyObject *py_keys, *py_np_metas;
    if (!PyArg_ParseTuple(args, "OO", &py_keys, &py_np_metas)) {
        return NULL;
    }

    if (py_np_metas == Py_None) {
        std::string error_msg = "The numpy metadatas can't be None";
        PyErr_SetString(PyExc_TypeError, error_msg.c_str());
        return NULL;
    }

    HArrayMetadata *np_metas = reinterpret_cast<HArrayMetadata *>(py_np_metas);

    const uint64_t *storage_id = parse_uuid(py_keys);
    PyObject *obj=Py_None;
    try {
        obj = self->NumpyDataStore->get_row_elements(storage_id, np_metas->np_metas);
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    delete[] storage_id;
    return obj;
}

static PyObject *get_cluster_ids(HNumpyStore *self, PyObject *args) {
    PyObject *py_np_metas;
    if (!PyArg_ParseTuple(args, "O", &py_np_metas)) {
        return NULL;
    }

    if (py_np_metas == Py_None) {
        std::string error_msg = "The numpy metadatas can't be None";
        PyErr_SetString(PyExc_TypeError, error_msg.c_str());
        return NULL;
    }

    HArrayMetadata *np_metas = reinterpret_cast<HArrayMetadata *>(py_np_metas);

    PyObject *result = Py_None;
    try {

        std::list<int32_t> clusters = self->NumpyDataStore->get_cluster_ids(np_metas->np_metas);

        uint16_t nclusters = clusters.size();
        result = PyList_New(nclusters);
        uint16_t key_i = 0;
        for (auto x : clusters) {
            PyList_SetItem(result, key_i++, PyLong_FromLong(x));
        }
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return result;
}

/* Returns a [ (zorder_id, cluster_id, block_id, (block_coordinates)) ] */
static PyObject *get_block_ids(HNumpyStore *self, PyObject *args) {
    PyObject *py_np_metas;
    if (!PyArg_ParseTuple(args, "O", &py_np_metas)) {
        return NULL;
    }

    if (py_np_metas == Py_None) {
        std::string error_msg = "The numpy metadatas can't be None";
        PyErr_SetString(PyExc_TypeError, error_msg.c_str());
        return NULL;
    }

    HArrayMetadata *np_metas = reinterpret_cast<HArrayMetadata *>(py_np_metas);

    PyObject *result = Py_None;
    try {

        std::list<std::tuple<uint64_t, uint32_t, uint32_t, std::vector<uint32_t>>> clusters;
        clusters = self->NumpyDataStore->get_block_ids(np_metas->np_metas);

        uint16_t nclusters = clusters.size();
        result = PyList_New(nclusters);
        uint16_t key_i = 0;
        for (auto x : clusters) {
            std::vector<uint32_t> ccs = std::get<3>(x);
            PyObject * pyccs = PyTuple_New( ccs.size() );
            uint32_t pos = 0;
            for(uint32_t id : ccs) {
                PyTuple_SetItem( pyccs, pos, PyLong_FromLong( id ) );
                pos ++;
            }
            PyObject * mitupla = PyTuple_Pack(4,
                                              PyLong_FromLong(std::get<0>(x)),
                                              PyLong_FromLong(std::get<1>(x)),
                                              PyLong_FromLong(std::get<2>(x)),
                                              pyccs);
            PyList_SetItem(result, key_i++, mitupla);
        }
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return result;
}

static PyObject *allocate_numpy(HNumpyStore *self, PyObject *args) {
    PyObject *py_keys, *py_np_metas;
    if (!PyArg_ParseTuple(args, "OO", &py_keys, &py_np_metas)) {
        return NULL;
    }

    if (py_np_metas == Py_None) {
        std::string error_msg = "The numpy metadatas can't be None";
        PyErr_SetString(PyExc_TypeError, error_msg.c_str());
        return NULL;
    }

    HArrayMetadata *np_metas = reinterpret_cast<HArrayMetadata *>(py_np_metas);

    const uint64_t *storage_id = parse_uuid(py_keys);
    PyObject *res;
    try {
        res = self->NumpyDataStore->reserve_numpy_space(storage_id, np_metas->np_metas);
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    delete[] storage_id;
    PyObject *result_list = PyList_New(1);
    PyList_SetItem(result_list, 0, res ? res : Py_None);
    return result_list;
}

/***
 * Receives a uuid, makes the reservation of the numpy specified in the storage_id and computes the number of elements inside each row of a block
 * @param self Python HNumpyStore object upon method invocation
 * @param args Arg tuple containing one list with the the keys. Keys are made of a list with a UUID and
 * values of a list with a single numpy ndarray.
 * @return A list with two elements: the first has the numpy memory reserved and the second has the row elements
 */
static PyObject *store_numpy_slices(HNumpyStore *self, PyObject *args) {
    int py_order;
    PyObject *py_keys, *py_numpy, *py_np_metas, *py_coord;
    if (!PyArg_ParseTuple(args, "OOOOi", &py_keys, &py_np_metas, &py_numpy, &py_coord, &py_order)) {
        return NULL;
    }

    for (uint16_t key_i = 0; key_i < PyList_Size(py_keys); ++key_i) {
        if (PyList_GetItem(py_keys, key_i) == Py_None) {
            std::string error_msg = "Keys can't be None, key_position: " + std::to_string(key_i);
            PyErr_SetString(PyExc_TypeError, error_msg.c_str());
            return NULL;
        }
    }

    if (PyList_Size(py_keys) != 1) {
        std::string error_msg = "Only one uuid as a key can be passed";
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    };

    if (PyList_Size(py_numpy) != 1) {
        std::string error_msg = "Only one numpy can be saved at once";
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    };

    if (py_np_metas == Py_None) {
        std::string error_msg = "The numpy metadatas can't be None";
        PyErr_SetString(PyExc_TypeError, error_msg.c_str());
        return NULL;
    }

    HArrayMetadata *np_metas = reinterpret_cast<HArrayMetadata *>(py_np_metas);

    const uint64_t *storage_id = parse_uuid(PyList_GetItem(py_keys, 0));

    // Transform the object to the numpy ndarray
    PyArrayObject *numpy_arr;
    if (!PyArray_OutputConverter(PyList_GetItem(py_numpy, 0), &numpy_arr)) {
        std::string error_msg = "Can't convert the given numpy to a numpy ndarray";
        PyErr_SetString(PyExc_TypeError, error_msg.c_str());
        return NULL;
    }

    try {
        self->NumpyDataStore->store_numpy(storage_id, np_metas->np_metas, numpy_arr, py_coord, py_order);
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    delete[] storage_id;
    Py_RETURN_NONE;
}

static PyObject *wait(HNumpyStore *self, PyObject *args) {
    try {
        self->NumpyDataStore->wait_stores();
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        std::string err_msg = "Waiting write of elements failed with " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, err_msg.c_str());
        return NULL;
    }

    Py_RETURN_NONE;
}

/***
 * Receives a uuid (storage_id for the numpy), metadatas, a pointer to the
 * numpy base address where data will be saved, and a list of coordinates/columns to
 * recover. The function will load the numpy columns in the memory at the
 * pointer according to the coordinates/columns.
 * @param self Python HNumpyStore object upon method invocation
 * @param args Arg tuple containing one list with the the keys.
 *             Keys are made of:
 *             - a list with a UUID,
 *             - an object with the numpy metadata information,
 *             - a list with a pointer reserved with the numpy size that will
 *               be used to store the numpy,
 *             - a list of column IDs which specifies the numpy columns to read or
 *               the list of coordinates which specifies the numpy chunk to read
 * @return None
 */
static PyObject *load_numpy_slices(HNumpyStore *self, PyObject *args) {
    //We need to include the numpy key in the parameters list, results -> reserved numpy

    int py_order;
    PyObject *py_keys, *py_store, *py_coord, *py_np_metas;
    if (!PyArg_ParseTuple(args, "OOOOi", &py_keys, &py_np_metas, &py_store, &py_coord, &py_order)) {
        return NULL;
    }

    // Only one uuid as a key
    if (PyList_Size(py_keys) != 1) {
        std::string error_msg = "Only one uuid as a key can be passed";
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    };

    // Only one numpy as a value
    if (PyList_Size(py_store) != 1) {
        std::string error_msg = "Only one numpy can be saved at once";
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    };

    if (py_np_metas == Py_None) {
        std::string error_msg = "The numpy metadatas can't be None";
        PyErr_SetString(PyExc_TypeError, error_msg.c_str());
        return NULL;
    }

    PyObject *numpy = PyList_GetItem(py_store, 0);
    if (numpy == Py_None) {
        std::string error_msg = "The numpy can't be None";
        PyErr_SetString(PyExc_TypeError, error_msg.c_str());
        return NULL;
    }

    // Transform the object to the numpy ndarray
    PyArrayObject *numpy_arr;
    if (!PyArray_OutputConverter(numpy, &numpy_arr)) {
        std::string error_msg = "Can't convert the given numpy to a numpy ndarray";
        PyErr_SetString(PyExc_TypeError, error_msg.c_str());
        return NULL;
    }
    const uint64_t *storage_id = parse_uuid(PyList_GetItem(py_keys, 0));
    HArrayMetadata *np_metas = reinterpret_cast<HArrayMetadata *>(py_np_metas);

    try {
        self->NumpyDataStore->load_numpy(storage_id, np_metas->np_metas, numpy_arr, py_coord, py_order);
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    delete[] storage_id;
    Py_RETURN_NONE;
}


static void hnumpy_store_dealloc(HNumpyStore *self) {
    delete (self->NumpyDataStore);
    Py_TYPE((PyObject *) self)->tp_free((PyObject *) self);
}


static PyObject *hnumpy_store_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    HNumpyStore *self;
    self = (HNumpyStore *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}


static int hnumpy_store_init(HNumpyStore *self, PyObject *args, PyObject *kwds) {

    const char *table, *keyspace;
    PyObject *py_config;

    if (!PyArg_ParseTuple(args, "ssO", &keyspace, &table, &py_config)) {
        return -1;
    };


    /** PARSE CONFIG **/

    std::map<std::string, std::string> config;

    if (PyDict_Check(py_config)) {
        PyObject *dict, *key, *value;;
        if (!PyArg_Parse(py_config, "O", &dict)) return -1;

        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)) {
            std::string conf_key(PyUnicode_AsUTF8(key));
            if (PyUnicode_Check(value)) {
                std::string conf_val(PyUnicode_AsUTF8(value));
                config[conf_key] = conf_val;
            }
            if (PyLong_Check(value)) {
                int32_t c_val = (int32_t) PyLong_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }
            if (PyBool_Check(value)) {
                bool c_val = PyObject_IsTrue(value);
                if (!c_val) config[conf_key] = "false";
                else config[conf_key] = "true";
            }
        }
    }

    /*** PARSE TABLE METADATA ***/

    try {
        self->NumpyDataStore = new NumpyStorage(table, keyspace, storage, config);
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    return 0;
}


static PyMethodDef hnumpy_store_type_methods[] = {
        {"allocate_numpy",       (PyCFunction) allocate_numpy,       METH_VARARGS, NULL},
        {"store_numpy_slices",   (PyCFunction) store_numpy_slices,   METH_VARARGS, NULL},
        {"wait",                 (PyCFunction) wait,                 METH_VARARGS, NULL},
        {"load_numpy_slices",    (PyCFunction) load_numpy_slices,    METH_VARARGS, NULL},
        {"get_elements_per_row", (PyCFunction) get_elements_per_row, METH_VARARGS, NULL},
        {"get_cluster_ids",      (PyCFunction) get_cluster_ids,      METH_VARARGS, NULL},
        {"get_block_ids",        (PyCFunction) get_block_ids,        METH_VARARGS, NULL},
        {NULL, NULL, 0,                                                            NULL}
};


static PyTypeObject hfetch_HNumpyStoreType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "hfetch.HNumpyStore",             /* tp_name */
        sizeof(HNumpyStore), /* tp_basicsize */
        0,                         /*tp_itemsize*/
        (destructor) hnumpy_store_dealloc, /*tp_dealloc*/
        0,                         /*tp_print*/
        0,                         /*tp_getattr*/
        0,                         /*tp_setattr*/
        0,                         /*tp_compare*/
        0,                         /*tp_repr*/
        0,                         /*tp_as_number*/
        0,                         /*tp_as_sequence*/
        0,                         /*tp_as_mapping*/
        0,                         /*tp_hash */
        0,                         /*tp_call*/
        0,                         /*tp_str*/
        0,                         /*tp_getattro*/
        0,                         /*tp_setattro*/
        0,                         /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
        "Cassandra rows",           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,                       /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        hnumpy_store_type_methods,             /* tp_methods */
        0,             /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc) hnumpy_store_init,      /* tp_init */
        0,                         /* tp_alloc */
        hnumpy_store_new,                 /* tp_new */
};

/*** Numpy metadata expose ****/


static PyObject *harray_metadata_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    HArrayMetadata *self;
    self = (HArrayMetadata *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}


static int harray_metadata_init(HArrayMetadata *self, PyObject *args, PyObject *kwds) {
    const char *kwlist[] = {"dims", "strides", "typekind", "byteorder", "elem_size", "flags", "partition_type", NULL};


    const char *typekind_tmp, *byteorder_tmp;
    self->np_metas = ArrayMetadata();
    PyObject *dims, *strides;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOssiib", (char **)kwlist, &dims, &strides,
                                     &typekind_tmp, &byteorder_tmp,
                                     &self->np_metas.elem_size, &self->np_metas.flags,
                                     &self->np_metas.partition_type)) {
        return -1;
    }


// Dims and ndims
    if (!PyList_Check(dims)) throw ModuleException("numpy metadata missing dims");
    int32_t ndims = PyList_Size(dims);
    self->np_metas.dims.resize(ndims);
    self->np_metas.strides.resize(ndims);


    for (int32_t dim_i = 0; dim_i < ndims; ++dim_i) {
        PyObject *elem_dim = PyList_GetItem(dims, dim_i);
        if (elem_dim == Py_None) throw ModuleException("numpy metadata missing dims");
        if (!PyLong_Check(elem_dim) || !PyArg_Parse(elem_dim, Py_INT, &self->np_metas.dims[dim_i]))
            throw ModuleException("Numpy dims must be a list of ints");
    }


// Strides
    if (!PyList_Check(strides)) throw ModuleException("numpy metadata missing strides");

    if (PyList_Size(strides) != ndims) throw ModuleException("Numpy strides must be a list of ints");
    for (int32_t dim_i = 0; dim_i < ndims; ++dim_i) {
        PyObject *elem_dim = PyList_GetItem(strides, dim_i);
        if (elem_dim == Py_None) throw ModuleException("numpy metadata missing strides");
        if (!PyLong_Check(elem_dim) || !PyArg_Parse(elem_dim, Py_INT, &self->np_metas.strides[dim_i]))
            throw ModuleException("Numpy strides must be a list of ints");
    }

    self->np_metas.typekind = typekind_tmp[0];
    self->np_metas.byteorder = byteorder_tmp[0];


    return 0;
}


static void harray_metadata_dealloc(HArrayMetadata *self) {
    Py_TYPE((PyObject *) self)->tp_free((PyObject *) self);
}

static PyObject *harray_metadata_repr(PyObject *self) {
    HArrayMetadata *array_metas = (HArrayMetadata *) self;
    std::string repr = "Typekind: " + std::to_string(array_metas->np_metas.typekind) + ", elem_size:" +
                       std::to_string(array_metas->np_metas.elem_size) + ", partition_type: " +
                       std::to_string(array_metas->np_metas.partition_type);
    repr += ", dims[ ";
    for(uint32_t i : array_metas->np_metas.dims) {
        repr += std::to_string(i) + " ";
    }
    repr += "], ";
    repr += "strides[ ";
    for(uint32_t i : array_metas->np_metas.strides) {
        repr += std::to_string(i) + " ";
    }
    repr += "], ";
    repr += "Flags: " + std::to_string(array_metas->np_metas.flags);
    repr += ", ";
    repr += "Byteorder: " + std::to_string(array_metas->np_metas.byteorder);

    PyObject *py_repr = PyUnicode_FromStringAndSize(repr.c_str(), repr.length());
    if (!py_repr) {
        std::string error = "Can't represent the numpy metadatas";
        PyErr_SetString(PyExc_RuntimeError, error.c_str());
        return NULL;
    }
    return py_repr;
}

static PyMemberDef harray_metadata_type_members[] = {
        {"elem_size", /* name */
                T_INT, /* type */
                offsetof(HArrayMetadata, np_metas.elem_size),  /* offset */
                0,  /* flags */
                NULL  /* docstring */},
        {"flags", /* name */
                T_INT, /* type */
                offsetof(HArrayMetadata, np_metas.flags),  /* offset */
                0,  /* flags */
                NULL  /* docstring */},
        {"partition_type", /* name */
                T_UBYTE, /* type */
                offsetof(HArrayMetadata, np_metas.partition_type),  /* offset */
                0,  /* flags */
                NULL  /* docstring */},
        {"typekind", /* name */
                T_CHAR, /* type */
                offsetof(HArrayMetadata, np_metas.typekind),  /* offset */
                0,  /* flags */
                NULL  /* docstring */},
        {"byteorder", /* name */
                T_CHAR, /* type */
                offsetof(HArrayMetadata, np_metas.byteorder),  /* offset */
                0,  /* flags */
                NULL  /* docstring */},
        {NULL},
};


static int register_harray(PyObject *self, PyObject *keyspace) {
    return -1;
}

static PyObject *get_strides(HArrayMetadata *self, void *closure) {
    size_t n_strides = self->np_metas.strides.size();
    PyObject *py_strides = PyList_New(n_strides);

    for (uint16_t i = 0; i < n_strides; i++) {
        PyList_SetItem(py_strides, i, Py_BuildValue(Py_INT, self->np_metas.strides[i]));
    }
    return py_strides;
}

static int set_strides(HArrayMetadata *self, PyObject *value, void *closure) {
    if (!PySequence_Check(value))
        return -1;
    self->np_metas.strides.clear();

    PyObject *iter = PySeqIter_New(value);
    PyObject *elem;
    while ((elem = PyIter_Next(iter)) != NULL) {
        uint32_t stride_i;
        if (!PyLong_Check(elem))
            return -1;
        PyArg_Parse(elem, Py_INT, &stride_i);
        self->np_metas.strides.push_back(stride_i);
    }
    return 0;
}


static PyObject *get_dims(HArrayMetadata *self, void *closure) {
    size_t n_dims = self->np_metas.dims.size();
    PyObject *py_strides = PyList_New(n_dims);

    for (uint16_t i = 0; i < n_dims; i++) {
        PyList_SetItem(py_strides, i, Py_BuildValue(Py_INT, self->np_metas.dims[i]));
    }
    return py_strides;
}

static int set_dims(HArrayMetadata *self, PyObject *value, void *closure) {
    if (!PySequence_Check(value))
        return -1;
    self->np_metas.dims.clear();

    PyObject *iter = PySeqIter_New(value);
    PyObject *elem;
    while ((elem = PyIter_Next(iter)) != NULL) {
        uint32_t stride_i;
        if (!PyLong_Check(elem))
            return -1;
        PyArg_Parse(elem, Py_INT, &stride_i);
        self->np_metas.dims.push_back(stride_i);
    }
    return 0;
}

static PyGetSetDef harray_metadata_getset_type[] = {
        {"strides", (getter) get_strides, (setter) set_strides, "strides attr", NULL},
        {"dims",    (getter) get_dims,    (setter) set_dims,    "dims attr",    NULL},
        {NULL} /* Sentinel */

};


static PyTypeObject hfetch_HArrayMetadataType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "hfetch.HArrayMetadata",             /* tp_name */
        sizeof(HArrayMetadata), /* tp_basicsize */
        0,                         /*tp_itemsize*/
        (destructor) harray_metadata_dealloc, /*tp_dealloc*/
        0,                         /*tp_print*/
        0,                         /*tp_getattr*/
        0,                         /*tp_setattr*/
        0,                         /*tp_compare*/
        harray_metadata_repr,                         /*tp_repr*/
        0,                         /*tp_as_number*/
        0,                         /*tp_as_sequence*/
        0,                         /*tp_as_mapping*/
        0,                         /*tp_hash */
        0,                         /*tp_call*/
        0,                         /*tp_str*/
        0,                         /*tp_getattro*/
        0,                         /*tp_setattro*/
        0,                         /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
        "Array metadata",           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,                       /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        0,             /* tp_methods */
        harray_metadata_type_members,             /* tp_members */
        harray_metadata_getset_type,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc) harray_metadata_init,      /* tp_init */
        0,                         /* tp_alloc */
        harray_metadata_new,                 /* tp_new */
};


/*** ITERATOR METHODS AND SETUP ***/


static PyObject *hiter_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    HIterator *self;
    self = (HIterator *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}


static int hiter_init(HIterator *self, PyObject *args, PyObject *kwds) {
    const char *table, *keyspace;

    PyObject *py_tokens, *py_keys_names, *py_cols_names, *py_config;
    if (!PyArg_ParseTuple(args, "ssOOOO", &keyspace, &table, &py_tokens,
                          &py_keys_names,
                          &py_cols_names, &py_config)) {
        return -1;
    };


    uint16_t tokens_size = (uint16_t) PyList_Size(py_tokens);
    uint16_t keys_size = (uint16_t) PyList_Size(py_keys_names);
    uint16_t cols_size = (uint16_t) PyList_Size(py_cols_names);

    self->token_ranges = std::vector<std::pair<int64_t, int64_t >>(tokens_size);
    for (uint16_t i = 0; i < tokens_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_tokens, i);
        int64_t t_a, t_b;
        if (!PyArg_ParseTuple(obj_to_convert, "LL", &t_a, &t_b)) {
            return -1;
        };

        self->token_ranges[i] = std::make_pair(t_a, t_b);
    }


    std::vector<std::map<std::string, std::string>> keys_names(keys_size);

    for (uint16_t i = 0; i < keys_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_keys_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        }
        keys_names[i] = {{"name", std::string(str_temp)}};
    }

    std::vector<std::map<std::string, std::string>> columns_names(cols_size);
    for (uint16_t i = 0; i < cols_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_cols_names, i);

        if (PyUnicode_Check(obj_to_convert)) {
            char *str_temp;
            if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
                return -1;
            };
            columns_names[i] = {{"name", std::string(str_temp)}};
        } else if (PyDict_Check(obj_to_convert)) {
            //CASE NUMPY
            PyObject *dict;
            if (!PyArg_Parse(obj_to_convert, "O", &dict)) {
                return -1;
            };

            PyObject *aux_table = PyDict_GetItemString(dict, "npy_table");
            if (aux_table != NULL) {
                columns_names[i]["npy_table"] = PyUnicode_AsUTF8(aux_table);
            }
            PyObject *py_name = PyDict_GetItemString(dict, "name");
            columns_names[i]["name"] = PyUnicode_AsUTF8(py_name);

            PyObject *py_arr_type = PyDict_GetItemString(dict, "type");
            columns_names[i]["type"] = PyUnicode_AsUTF8(py_arr_type);

            PyObject *py_arr_dims = PyDict_GetItemString(dict, "dims");
            columns_names[i]["dims"] = PyUnicode_AsUTF8(py_arr_dims);

            PyObject *py_arr_partition = PyDict_GetItemString(dict, "partition");
            if (std::strcmp(PyUnicode_AsUTF8(py_arr_partition), "true") == 0) {
                columns_names[i]["partition"] = "partition";
            } else columns_names[i]["partition"] = "no-partition";
        } else {
            PyErr_SetString(PyExc_TypeError, "Can't parse column names, expected String, Dict or Unicode");
            return -1;
        }
    }





    /** PARSE CONFIG **/

    std::map<std::string, std::string> config;
    int type_check = PyDict_Check(py_config);
    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return -1;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)) {
            std::string conf_key(PyUnicode_AsUTF8(key));
            if (PyUnicode_Check(value)) {
                std::string conf_val(PyUnicode_AsUTF8(value));
                config[conf_key] = conf_val;
            }
            if (PyLong_Check(value)) {
                int32_t c_val = (int32_t) PyLong_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }
        }
    }

    try {
        self->P = storage->get_iterator(table, keyspace, keys_names, columns_names, self->token_ranges, config);
        if (self->P->get_type() == "items") {
            self->rowParser = new PythonParser(storage, self->P->get_metadata()->get_items());
        } else if (self->P->get_type() == "values")
            self->rowParser = new PythonParser(storage, self->P->get_metadata()->get_values());
        else self->rowParser = new PythonParser(storage, self->P->get_metadata()->get_keys());
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }

    return 0;
}


static PyObject *get_next(HIterator *self) {
    const TupleRow *result;
    try {
        result = self->P->get_cnext();
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    if (!result) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    std::vector<const TupleRow *> temp = {result};

    PyObject *py_row;
    try {
        py_row = self->rowParser->make_pylist(temp);
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        std::string error_msg = "Get next, parse result: " + std::string(e.what());
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    }
    delete (result);
    return py_row;
}

static void hiter_dealloc(HIterator *self) {
    if (self->rowParser) delete (self->rowParser);
    if (self->P) delete (self->P);
    Py_TYPE((PyObject *) self)->tp_free((PyObject *) self);
}


static PyMethodDef hiter_type_methods[] = {
        {"get_next", (PyCFunction) get_next, METH_NOARGS, NULL},
        {NULL, NULL, 0,                                   NULL}
};


static PyTypeObject hfetch_HIterType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "hfetch.HIter",             /* tp_name */
        sizeof(HIterator), /* tp_basicsize */
        0,                         /*tp_itemsize*/
        (destructor) hiter_dealloc, /*tp_dealloc*/
        0,                         /*tp_print*/
        0,                         /*tp_getattr*/
        0,                         /*tp_setattr*/
        0,                         /*tp_compare*/
        0,                         /*tp_repr*/
        0,                         /*tp_as_number*/
        0,                         /*tp_as_sequence*/
        0,                         /*tp_as_mapping*/
        0,                         /*tp_hash */
        0,                         /*tp_call*/
        0,                         /*tp_str*/
        0,                         /*tp_getattro*/
        0,                         /*tp_setattro*/
        0,                         /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
        "Cassandra iter",           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,                       /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        hiter_type_methods,             /* tp_methods */
        0,             /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc) hiter_init,      /* tp_init */
        0,                         /* tp_alloc */
        hiter_new,                 /* tp_new */
};

/*** WRITER METHODS ***/


static PyObject *write_cass(HWriter *self, PyObject *args) {
    PyObject *py_keys, *py_values;
    if (!PyArg_ParseTuple(args, "OO", &py_keys, &py_values)) {
        return NULL;
    }

    try {
        TupleRow *k = self->keysParser->make_tuple(py_keys);
        TupleRow *v = self->valuesParser->make_tuple(py_values);
        self->W->write_to_cassandra(k, v);
        delete (k);
        delete (v);
    }
    catch (TypeErrorException &e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    Py_RETURN_NONE;
}


static PyObject *hwriter_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    HWriter *self;
    self = (HWriter *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}

static int hwriter_init(HWriter *self, PyObject *args, PyObject *kwds) {
    const char *table, *keyspace;
    PyObject *py_keys_names, *py_cols_names, *py_config;
    if (!PyArg_ParseTuple(args, "ssOOO", &keyspace, &table,
                          &py_keys_names,
                          &py_cols_names, &py_config)) {
        return -1;
    };


    uint16_t keys_size = (uint16_t) PyList_Size(py_keys_names);
    uint16_t cols_size = (uint16_t) PyList_Size(py_cols_names);

    std::vector<std::map<std::string, std::string>> keys_names(keys_size);

    for (uint16_t i = 0; i < keys_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_keys_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        }
        keys_names[i] = {{"name", std::string(str_temp)}};
    }

    std::vector<std::map<std::string, std::string>> columns_names(cols_size);
    for (uint16_t i = 0; i < cols_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_cols_names, i);

        if (PyUnicode_Check(obj_to_convert)) {
            char *str_temp;
            if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
                return -1;
            };
            columns_names[i] = {{"name", std::string(str_temp)}};
        } else if (PyDict_Check(obj_to_convert)) {
            //CASE NUMPY
            PyObject *dict;
            if (!PyArg_Parse(obj_to_convert, "O", &dict)) {
                return -1;
            };

            PyObject *aux_table = PyDict_GetItemString(dict, "npy_table");
            if (aux_table != NULL) {
                columns_names[i]["npy_table"] = PyUnicode_AsUTF8(aux_table);
            }
            PyObject *py_name = PyDict_GetItemString(dict, "name");
            columns_names[i]["name"] = PyUnicode_AsUTF8(py_name);

            PyObject *py_arr_type = PyDict_GetItemString(dict, "type");
            columns_names[i]["type"] = PyUnicode_AsUTF8(py_arr_type);

            PyObject *py_arr_dims = PyDict_GetItemString(dict, "dims");
            columns_names[i]["dims"] = PyUnicode_AsUTF8(py_arr_dims);

            PyObject *py_arr_partition = PyDict_GetItemString(dict, "partition");
            if (std::strcmp(PyUnicode_AsUTF8(py_arr_partition), "true") == 0) {
                columns_names[i]["partition"] = "partition";
            } else columns_names[i]["partition"] = "no-partition";
        } else {
            PyErr_SetString(PyExc_TypeError, "Can't parse column names, expected String, Dict or Unicode");
            return -1;
        }
    }
    /** PARSE CONFIG **/

    std::map<std::string, std::string> config;
    int type_check = PyDict_Check(py_config);
    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return -1;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)) {
            std::string conf_key(PyUnicode_AsUTF8(key));
            if (PyUnicode_Check(value)) {
                std::string conf_val(PyUnicode_AsUTF8(value));
                config[conf_key] = conf_val;
            }
            if (PyLong_Check(value)) {
                int32_t c_val = (int32_t) PyLong_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }
            if (PyBool_Check(value)) {
                int truthy = PyObject_IsTrue(value);
                if (truthy) config[conf_key] = "True";
                else config[conf_key] = "False";

            }
        }
    }
    try {
        self->W = storage->make_writer(table, keyspace, keys_names, columns_names, config);
        self->keysParser = new PythonParser(storage, self->W->get_metadata()->get_keys());
        self->valuesParser = new PythonParser(storage, self->W->get_metadata()->get_values());
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }

    return 0;
}

static void hwriter_dealloc(HWriter *self) {
    if (self->keysParser) delete (self->keysParser);
    if (self->valuesParser) delete (self->valuesParser);
    if (self->W) delete (self->W);
    Py_TYPE((PyObject *) self)->tp_free((PyObject *) self);
}


static PyMethodDef hwriter_type_methods[] = {
        {"write", (PyCFunction) write_cass, METH_VARARGS, NULL},
        {NULL, NULL, 0,                                   NULL}
};

static PyTypeObject hfetch_HWriterType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "hfetch.HWriter",             /* tp_name */
        sizeof(HWriter), /* tp_basicsize */
        0,                         /*tp_itemsize*/
        (destructor) hwriter_dealloc, /*tp_dealloc*/
        0,                         /*tp_print*/
        0,                         /*tp_getattr*/
        0,                         /*tp_setattr*/
        0,                         /*tp_compare*/
        0,                         /*tp_repr*/
        0,                         /*tp_as_number*/
        0,                         /*tp_as_sequence*/
        0,                         /*tp_as_mapping*/
        0,                         /*tp_hash */
        0,                         /*tp_call*/
        0,                         /*tp_str*/
        0,                         /*tp_getattro*/
        0,                         /*tp_setattro*/
        0,                         /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
        "Cassandra writer",           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,                       /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        hwriter_type_methods,             /* tp_methods */
        0,             /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc) hwriter_init,      /* tp_init */
        0,                         /* tp_alloc */
        hwriter_new,                 /* tp_new */
};


/*** MODULE SETUP ****/



static PyMethodDef module_methods[] = {
        {"connectCassandra",    (PyCFunction) connectCassandra,    METH_VARARGS, NULL},
        {"disconnectCassandra", (PyCFunction) disconnectCassandra, METH_NOARGS,  NULL},
        {NULL, NULL, 0,                                                          NULL}
};


static PyObject *create_iter_items(HCache *self, PyObject *args) {
    PyObject *py_config;
    if (!PyArg_ParseTuple(args, "O", &py_config)) {
        return NULL;
    }

    std::map<std::string, std::string> config;
    int type_check = PyDict_Check(py_config);

    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return NULL;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)) {
            std::string conf_key(PyUnicode_AsUTF8(key));
            if (PyUnicode_Check(value)) {
                std::string conf_val(PyUnicode_AsUTF8(value));
                config[conf_key] = conf_val;
            }
            if (PyLong_Check(value)) {
                int32_t c_val = (int32_t) PyLong_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }
        }
    } else if (PyLong_Check((py_config))) {
        int32_t c_val = (int32_t) PyLong_AsLong(py_config);
        config["prefetch_size"] = std::to_string(c_val);
    }
    config["type"] = "items";

    HIterator *iter = (HIterator *) hiter_new(&hfetch_HIterType, args, args);

    //hiter_init(iter, args, args);
    if (!self->T) {
        PyErr_SetString(PyExc_RuntimeError, "Tried to create iteritems, but the cache didn't exist");
        return NULL;
    }

    // Flush elements to avoid coherency problems
    self->T->flush_elements();

    try {
        iter->P = storage->get_iterator(self->T->get_metadata(), self->token_ranges, config);
        iter->rowParser = new PythonParser(storage, iter->P->get_metadata()->get_items());
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    return (PyObject *) iter;
}


static PyObject *create_iter_keys(HCache *self, PyObject *args) {
    PyObject *py_config;
    if (!PyArg_ParseTuple(args, "O", &py_config)) {
        return NULL;
    }

    std::map<std::string, std::string> config;
    int type_check = PyDict_Check(py_config);

    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return NULL;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)) {
            std::string conf_key(PyUnicode_AsUTF8(key));
            if (PyUnicode_Check(value)) {
                std::string conf_val(PyUnicode_AsUTF8(value));
                config[conf_key] = conf_val;
            }
            if (PyLong_Check(value)) {
                int32_t c_val = (int32_t) PyLong_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }

        }
    } else if (PyLong_Check((py_config))) {
        int32_t c_val = (int32_t) PyLong_AsLong(py_config);
        config["prefetch_size"] = std::to_string(c_val);
    }
    config["type"] = "keys";

    HIterator *iter = (HIterator *) hiter_new(&hfetch_HIterType, args, args);

    //hiter_init(iter, args, args);
    if (!self->T) {
        PyErr_SetString(PyExc_RuntimeError, "Tried to create iterkeys, but the cache didn't exist");
        return NULL;
    }
    try {
        iter->P = storage->get_iterator(self->T->get_metadata(), self->token_ranges, config);
        iter->rowParser = new PythonParser(storage, iter->P->get_metadata()->get_keys());
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    return (PyObject *) iter;
}


static PyObject *create_iter_values(HCache *self, PyObject *args) {

    PyObject *py_config;
    if (!PyArg_ParseTuple(args, "O", &py_config)) {
        return NULL;
    }

    std::map<std::string, std::string> config;
    int type_check = PyDict_Check(py_config);

    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return NULL;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)) {
            std::string conf_key(PyUnicode_AsUTF8(key));
            if (PyUnicode_Check(value)) {
                std::string conf_val(PyUnicode_AsUTF8(value));
                config[conf_key] = conf_val;
            }
            if (PyLong_Check(value)) {
                int32_t c_val = (int32_t) PyLong_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }

        }
    } else if (PyLong_Check((py_config))) {
        int32_t c_val = (int32_t) PyLong_AsLong(py_config);
        config["prefetch_size"] = std::to_string(c_val);
    }
    config["type"] = "values";

    HIterator *iter = (HIterator *) hiter_new(&hfetch_HIterType, args, args);

    //hiter_init(iter, args, args);
    if (!self->T) {
        PyErr_SetString(PyExc_RuntimeError, "Tried to create itervalues, but the cache didn't exist");
        return NULL;
    }
    try {
        iter->P = storage->get_iterator(self->T->get_metadata(), self->token_ranges, config);
        iter->rowParser = new PythonParser(storage, iter->P->get_metadata()->get_values());
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    return (PyObject *) iter;
}

void (*f)(PyObject *) = NULL;

static void module_dealloc(PyObject *self) {
    if (f) f(self);
}


static struct PyModuleDef hfetch_module_info = {
        PyModuleDef_HEAD_INIT,
        "hfetch",   /* name of module */
        nullptr, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        module_methods
};


PyMODINIT_FUNC
PyInit_hfetch(void) {

#define IMPORT_ERROR NULL
    hfetch_HNumpyStoreType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&hfetch_HNumpyStoreType) < 0)
        return IMPORT_ERROR;

    Py_INCREF(&hfetch_HNumpyStoreType);

    hfetch_HIterType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&hfetch_HIterType) < 0)
        return IMPORT_ERROR;

    Py_INCREF(&hfetch_HIterType);


    hfetch_HWriterType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&hfetch_HWriterType) < 0)
        return IMPORT_ERROR;

    Py_INCREF(&hfetch_HWriterType);


    hfetch_HCacheType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&hfetch_HCacheType) < 0)
        return IMPORT_ERROR;

    Py_INCREF(&hfetch_HCacheType);


    hfetch_HArrayMetadataType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&hfetch_HArrayMetadataType) < 0)
        return IMPORT_ERROR;

    Py_INCREF(&hfetch_HArrayMetadataType);

    PyObject *m = PyModule_Create(&hfetch_module_info);
    f = m->ob_type->tp_dealloc;
    m->ob_type->tp_dealloc = module_dealloc;

    PyModule_AddObject(m, "Hcache", (PyObject *) &hfetch_HCacheType);
    PyModule_AddObject(m, "HIterator", (PyObject *) &hfetch_HIterType);
    PyModule_AddObject(m, "HWriter", (PyObject *) &hfetch_HWriterType);
    PyModule_AddObject(m, "HNumpyStore", (PyObject *) &hfetch_HNumpyStoreType);
    PyModule_AddObject(m, "HArrayMetadata", (PyObject *) &hfetch_HArrayMetadataType);

    if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
    }
    return m;
}
