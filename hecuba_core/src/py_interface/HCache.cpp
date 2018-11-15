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

    self->ob_type->tp_free((PyObject *) self);
}


static PyObject *hcache_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    HCache *self;
    self = (HCache *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}


static std::string get_dict_item_as_text(PyObject* dict, std::string key) {
    PyObject* type = PyDict_GetItemString(dict, key.c_str());
    if (type) {
        std::string conf_val="";
        if (PyString_Check(type)) {
            char *buffer;
            size_t len;
            Py_ssize_t py_len;
            if (PyString_AsStringAndSize(type, &buffer, &py_len)) {
                PyObject* py_long = PyLong_FromSsize_t(py_len);
                len = PyLong_AsLong(py_long);
                return std::string(buffer, len);
            }
            return std::string();
        }
        else if (PyInt_Check(type)) {
            int32_t result = PyInt_AsLong(type);
            if (result==-1 && PyErr_Occurred()) return NULL;
            return std::to_string(result);
        }
    }
    return std::string();
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
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)) {
                std::string conf_val(PyString_AsString(value));
                config[conf_key] = conf_val;
            }
            if (PyInt_Check(value)) {
                int32_t c_val = (int32_t) PyInt_AsLong(value);
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

        if (PyString_Check(obj_to_convert) || PyUnicode_Check(obj_to_convert)) {
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
            columns_names[i]["name"] = PyString_AsString(py_name);

            if (PyObject *type = PyDict_GetItemString(dict, "type")) {
                if (std::strcmp(PyString_AsString(type), "numpy") == 0) {
                    columns_names[i]["table"] = std::string(table);
                    columns_names[i]["keyspace"] = std::string(keyspace);
                    columns_names[i]["numpy"] = "true";
                    columns_names[i]["type"] = "numpy";

                    std::string conf_val = get_dict_item_as_text(dict, "write_buffer_size");
                    if (!conf_val.empty()) columns_names[i]["write_buffer_size"] = conf_val;
                    conf_val = get_dict_item_as_text(dict, "write_callbacks_number");
                    if (!conf_val.empty()) columns_names[i]["write_callbacks_number"] = conf_val;

                    if (!PyByteArray_Check(py_storage_id)) {
                        //Object is UUID python class
                        uint32_t len = sizeof(uint64_t) * 2;
                        uint64_t *uuid = (uint64_t *) malloc(len);

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

                        columns_names[i]["storage_id"] = std::string((char *) uuid, len);
                        free(uuid);
                    } else {
                        uint32_t len = sizeof(uint64_t) * 2;
                        uint32_t len_found = (uint32_t) PyByteArray_Size(py_storage_id);
                        if (len_found != len) {
                            std::string error_msg = "UUID received has size " + std::to_string(len_found) +
                                                    ", expected was: " + std::to_string(len);
                            PyErr_SetString(PyExc_ValueError, error_msg.c_str());
                        }

                        char *cpp_bytes = PyByteArray_AsString(py_storage_id);

                        columns_names[i]["storage_id"] = std::string(cpp_bytes, len);
                    }
                }
            }
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

        if (PyString_Check(obj_to_convert) || PyUnicode_Check(obj_to_convert)) {
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

            PyObject *aux_table = PyDict_GetItem(dict, PyString_FromString("npy_table"));
            if (aux_table != NULL) {
                columns_names[i]["npy_table"] = PyString_AsString(aux_table);
            }
            PyObject *py_name = PyDict_GetItem(dict, PyString_FromString("name"));
            columns_names[i]["name"] = PyString_AsString(py_name);

            PyObject *py_arr_type = PyDict_GetItem(dict, PyString_FromString("type"));
            columns_names[i]["type"] = PyString_AsString(py_arr_type);

            PyObject *py_arr_dims = PyDict_GetItem(dict, PyString_FromString("dims"));
            columns_names[i]["dims"] = PyString_AsString(py_arr_dims);

            PyObject *py_arr_partition = PyDict_GetItem(dict, PyString_FromString("partition"));
            if (std::strcmp(PyString_AsString(py_arr_partition), "true") == 0) {
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
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)) {
                std::string conf_val(PyString_AsString(value));
                config[conf_key] = conf_val;
            }
            if (PyInt_Check(value)) {
                int32_t c_val = (int32_t) PyInt_AsLong(value);
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
    self->ob_type->tp_free((PyObject *) self);
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

        if (PyString_Check(obj_to_convert) || PyUnicode_Check(obj_to_convert)) {
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

            PyObject *aux_table = PyDict_GetItem(dict, PyString_FromString("npy_table"));
            if (aux_table != NULL) {
                columns_names[i]["npy_table"] = PyString_AsString(aux_table);
            }
            PyObject *py_name = PyDict_GetItem(dict, PyString_FromString("name"));
            columns_names[i]["name"] = PyString_AsString(py_name);

            PyObject *py_arr_type = PyDict_GetItem(dict, PyString_FromString("type"));
            columns_names[i]["type"] = PyString_AsString(py_arr_type);

            PyObject *py_arr_dims = PyDict_GetItem(dict, PyString_FromString("dims"));
            columns_names[i]["dims"] = PyString_AsString(py_arr_dims);

            PyObject *py_arr_partition = PyDict_GetItem(dict, PyString_FromString("partition"));
            if (std::strcmp(PyString_AsString(py_arr_partition), "true") == 0) {
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
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)) {
                std::string conf_val(PyString_AsString(value));
                config[conf_key] = conf_val;
            }
            if (PyInt_Check(value)) {
                int32_t c_val = (int32_t) PyInt_AsLong(value);
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
    self->ob_type->tp_free((PyObject *) self);
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
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)) {
                std::string conf_val(PyString_AsString(value));
                config[conf_key] = conf_val;
            }
            if (PyInt_Check(value)) {
                int32_t c_val = (int32_t) PyInt_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }
        }
    } else if PyInt_Check((py_config)) {
        int32_t c_val = (int32_t) PyInt_AsLong(py_config);
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
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)) {
                std::string conf_val(PyString_AsString(value));
                config[conf_key] = conf_val;
            }
            if (PyInt_Check(value)) {
                int32_t c_val = (int32_t) PyInt_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }

        }
    } else if PyInt_Check((py_config)) {
        int32_t c_val = (int32_t) PyInt_AsLong(py_config);
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
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)) {
                std::string conf_val(PyString_AsString(value));
                config[conf_key] = conf_val;
            }
            if (PyInt_Check(value)) {
                int32_t c_val = (int32_t) PyInt_AsLong(value);
                config[conf_key] = std::to_string(c_val);
            }

        }
    } else if PyInt_Check((py_config)) {
        int32_t c_val = (int32_t) PyInt_AsLong(py_config);
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

PyMODINIT_FUNC
inithfetch(void) {
    PyObject *m;
    hfetch_HIterType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&hfetch_HIterType) < 0)
        return;

    Py_INCREF(&hfetch_HIterType);


    hfetch_HWriterType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&hfetch_HWriterType) < 0)
        return;

    Py_INCREF(&hfetch_HWriterType);


    hfetch_HCacheType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&hfetch_HCacheType) < 0)
        return;

    Py_INCREF(&hfetch_HCacheType);


    m = Py_InitModule3("hfetch", module_methods, "c++ bindings for hecuba cache & prefetch");
    f = m->ob_type->tp_dealloc;
    m->ob_type->tp_dealloc = module_dealloc;

    PyModule_AddObject(m, "Hcache", (PyObject *) &hfetch_HCacheType);
    PyModule_AddObject(m, "HIterator", (PyObject *) &hfetch_HIterType);
    PyModule_AddObject(m, "HWriter", (PyObject *) &hfetch_HWriterType);
    if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        NUMPY_IMPORT_ARRAY_RETVAL;
    }
}
