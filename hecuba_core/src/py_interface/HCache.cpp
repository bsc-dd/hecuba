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


static void hcache_dealloc(HCache *self) {
    delete (self->keysParser);
    delete (self->valuesParser);
    delete (self->T);
    Py_TYPE((PyObject*) self)->tp_free((PyObject *) self);
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
        {"iterkeys",   (PyCFunction) create_iter_keys,   METH_VARARGS, NULL},
        {"itervalues", (PyCFunction) create_iter_values, METH_VARARGS, NULL},
        {"iteritems",  (PyCFunction) create_iter_items,  METH_VARARGS, NULL},
        {NULL, NULL, 0,                                                NULL}
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

        uint64_t time_low = (uint32_t) PyLong_AsLongLong(bytes);

        bytes = PyObject_GetAttrString(py_storage_id, "time_mid"); //16b
        uint64_t time_mid = (uint16_t) PyLong_AsLongLong(bytes);

        bytes = PyObject_GetAttrString(py_storage_id, "time_hi_version"); //16b
        uint64_t time_hi_version = (uint16_t) PyLong_AsLongLong(bytes);

        *uuid = (time_hi_version << 48) + (time_mid << 32) + (time_low);

        bytes = PyObject_GetAttrString(py_storage_id, "clock_seq_hi_variant"); //8b
        uint64_t clock_seq_hi_variant = (uint64_t) PyLong_AsLongLong(bytes);
        bytes = PyObject_GetAttrString(py_storage_id, "clock_seq_low"); //8b
        uint64_t clock_seq_low = (uint64_t) PyLong_AsLongLong(bytes);
        bytes = PyObject_GetAttrString(py_storage_id, "node"); //48b


        *(uuid + 1) = (uint64_t) PyLong_AsLongLong(bytes);
        *(uuid + 1) += clock_seq_hi_variant << 56;
        *(uuid + 1) += clock_seq_low << 48;

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


/***
 * Receives a numpy ndarray and a uuid, saves both to the table and keyspace passed during initialization
 * @param self Python HNumpyStore object upon method invocation
 * @param args Arg tuple containing two lists, the keys, and values. Keys are made of a list with a UUID,
 * values of a list with a single numpy ndarray.
 */
static PyObject *save_numpy(HNumpyStore *self, PyObject *args) {
    PyObject *py_keys, *py_values;
    if (!PyArg_ParseTuple(args, "OO", &py_keys, &py_values)) {
        return NULL;
    }

    // Only one uuid as a key
    if (PyList_Size(py_keys) != 1) {
        std::string error_msg = "Only one uuid as a key can be passed";
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    };

    // Only one numpy as a value
    if (PyList_Size(py_values) != 1) {
        std::string error_msg = "Only one numpy can be saved at once";
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    };


    for (uint16_t key_i = 0; key_i < PyList_Size(py_keys); ++key_i) {
        if (PyList_GetItem(py_keys, key_i) == Py_None) {
            std::string error_msg = "Keys can't be None, key_position: " + std::to_string(key_i);
            PyErr_SetString(PyExc_TypeError, error_msg.c_str());
            return NULL;
        }
    }


    const uint64_t *storage_id = parse_uuid(PyList_GetItem(py_keys, 0));

    PyObject *numpy = PyList_GetItem(py_values, 0);
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

    // 1 Extract metadatas && write data
    try {
        self->NumpyDataStore->store_numpy(storage_id, numpy_arr);
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *get_reserved_numpy(HNumpyStore *self, PyObject *args) {
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

    // Only one uuid as a key
    if (PyList_Size(py_keys) != 1) {
        std::string error_msg = "Only one uuid as a key can be passed";
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    };

    const uint64_t *storage_id = parse_uuid(PyList_GetItem(py_keys, 0));

    PyObject *numpy;
    try{
        numpy = self->NumpyDataStore->reserve_numpy_space(storage_id);
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    return numpy;
}

static PyObject *get_numpy(HNumpyStore *self, PyObject *args) {
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

    // Only one uuid as a key
    if (PyList_Size(py_keys) != 1) {
        std::string error_msg = "Only one uuid as a key can be passed";
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return NULL;
    };


    const uint64_t *storage_id = parse_uuid(PyList_GetItem(py_keys, 0));

    PyObject *numpy;
    try{
        numpy = self->NumpyDataStore->read_numpy(storage_id);
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    // Wrap the numpy into a list to follow the standard format of Hecuba
    PyObject *result_list = PyList_New(1);
    PyList_SetItem(result_list, 0, numpy ? numpy : Py_None);
    return result_list;
}

static PyObject *get_numpy_from_coordinates(HNumpyStore *self, PyObject *args) {
    //We need to include the numpy key in the parameters list, results -> reserved numpy
    PyObject *coord;
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
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &coord)) {
        PyErr_SetString(PyExc_TypeError, "Parameter must be a list");
        return NULL;
    }

    const uint64_t *storage_id = parse_uuid(PyList_GetItem(py_keys, 0));

    PyObject *reserved_array; //We need to include the reserved array in the parameters of the function, not here
    try{
        reserved_array = self->NumpyDataStore->coord_list_to_numpy(storage_id, coord); //i suppose we need storageid as the identifier of the data
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    PyObject *result_list = PyList_New(1);
    PyList_SetItem(result_list, 0, reserved_array ? reserved_array : Py_None);
    return result_list;
}




static void hnumpy_store_dealloc(HNumpyStore *self) {
    delete(self->NumpyDataStore);
    delete(self->cache);
    delete(self->read_cache);
    Py_TYPE((PyObject*) self)->tp_free((PyObject *) self);
}


static PyObject *hnumpy_store_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    HNumpyStore *self;
    self = (HNumpyStore *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}


static int hnumpy_store_init(HNumpyStore *self, PyObject *args, PyObject *kwds) {
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
        self->cache = storage->make_cache(table, keyspace, keys_names, columns_names, config);
        std::vector<std::map<std::string, std::string>> read_keys_names(keys_names.begin(), (keys_names.end()-1));
        std::vector<std::map<std::string, std::string>> read_columns_names = columns_names;


        read_columns_names.insert(read_columns_names.begin(),keys_names.back());


        self->read_cache = storage->make_cache(table, keyspace, read_keys_names, read_columns_names, config);


        self->NumpyDataStore = new NumpyStorage(self->cache, self->read_cache, config);
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    return 0;
}


static PyMethodDef hnumpy_store_type_methods[] = {
        {"get_numpy",  (PyCFunction) get_numpy,  METH_VARARGS, NULL},
        {"save_numpy", (PyCFunction) save_numpy, METH_VARARGS, NULL},
        {"get_reserved_numpy", (PyCFunction) get_reserved_numpy, METH_VARARGS, NULL},
        {"get_numpy_from_coordinates", (PyCFunction) get_numpy_from_coordinates, METH_VARARGS, NULL},
        {NULL, NULL, 0,                                        NULL}
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
    Py_TYPE((PyObject*) self)->tp_free((PyObject *) self);
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
    Py_TYPE((PyObject*) self)->tp_free((PyObject *) self);
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
        {"connectCassandra", (PyCFunction) connectCassandra, METH_VARARGS, NULL},
        {"disconnectCassandra", (PyCFunction) disconnectCassandra,  METH_NOARGS, NULL},
        {NULL, NULL, 0,                                                    NULL}
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
    } else if PyLong_Check((py_config)) {
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
    } else if PyLong_Check((py_config)) {
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
    } else if PyLong_Check((py_config)) {
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



    PyObject *m = PyModule_Create(&hfetch_module_info);
    f = m->ob_type->tp_dealloc;
    m->ob_type->tp_dealloc = module_dealloc;

    PyModule_AddObject(m, "Hcache", (PyObject *) &hfetch_HCacheType);
    PyModule_AddObject(m, "HIterator", (PyObject *) &hfetch_HIterType);
    PyModule_AddObject(m, "HWriter", (PyObject *) &hfetch_HWriterType);
    PyModule_AddObject(m, "HNumpyStore", (PyObject *) &hfetch_HNumpyStoreType);
    if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
    }
    return m;
}
