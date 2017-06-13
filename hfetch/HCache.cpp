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
            PyErr_SetString(PyExc_RuntimeError, "invalid contact point");
            return NULL;
        };
        contact_points += std::string(str_temp) + ",";
    }

    try {
        storage = std::make_shared<StorageInterface>(nodePort, contact_points);
        //storage = new StorageInterface(nodePort, contact_points);
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

    TupleRow *k;
    try {
        k = parser.make_tuple(py_keys, self->T->get_metadata()->get_keys());
    }
    catch (ModuleException e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    if (self->has_numpy) {
        // check partition TODO
        uint16_t numpy_pos = 0;
        std::shared_ptr<const std::vector<ColumnMeta> > metas = self->T->get_metadata()->get_values();
        while (parser.get_arr_type(metas->at(numpy_pos)) == NPY_NOTYPE && numpy_pos < metas->size()) ++numpy_pos;
        if (numpy_pos == metas->size()) {
            PyErr_SetString(PyExc_RuntimeError, "Sth went wrong looking for the numpy");
            return NULL;
        }
        //external
        if (metas->at(numpy_pos).info.size() == 5) {
            /*** PREPARE WRITER FOR AUXILIARY TABLE ***/
            std::map<std::string, std::string> config;
            std::vector<std::map<std::string, std::string> > numpy_columns(2);
            std::vector<std::map<std::string, std::string> > numpy_keys(1,{{"name","uuid"}});
            numpy_columns[0] = {{"name","data"}, {"type",(metas->at(numpy_pos).info.find("type")->second)},
                                {"dims",std::string(metas->at(numpy_pos).info.find("dims")->second)},
                                {"partition",std::string(metas->at(numpy_pos).info.find("partition")->second)}};
            numpy_columns[1] = {{"name","position"}};
            Writer *temp = NULL;
            try {
                temp = storage->make_writer(metas->at(numpy_pos).info.find("npy_table")->second.c_str(), self->T->get_metadata()->get_keyspace(),
                                            numpy_keys, numpy_columns, config);
            } catch (ModuleException e) {
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return NULL;
            }

            /*** Retrieve numpy array ***/

            PyObject *npy_list = PyList_New(1);

            PyObject *array = PyList_GetItem(py_values, numpy_pos);

            PyList_SetItem(npy_list, 0, array);
            Py_INCREF(array);


            std::vector<const TupleRow *> value_list;
            try {
                value_list = parser.make_tuples_with_npy(npy_list, temp->get_metadata()->get_values());
            }
            catch (std::exception &e) {
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return NULL;
            }
            Py_DECREF(npy_list);

            /*** Generate UUID ***/
            CassUuid uuid;
            CassUuidGen *uuid_gen = cass_uuid_gen_new();
            cass_uuid_gen_random(uuid_gen, &uuid);
            cass_uuid_gen_free(uuid_gen);

            uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);
            *c_uuid = uuid.time_and_version;
            *(c_uuid + 1) = uuid.clock_seq_and_node;

            void *payload = malloc(sizeof(uint64_t *));
            memcpy(payload, &c_uuid, sizeof(uint64_t *));

            /*** Store array ***/
            TupleRow *numpy_key = new TupleRow(temp->get_metadata()->get_keys(), sizeof(uint64_t) * 2, payload);

            for (const TupleRow *T:value_list) {
                TupleRow *key_copy = new TupleRow(numpy_key);
                temp->write_to_cassandra(key_copy, T);
            }

            /*** Store UUID ***/

            PyObject *py_uuid = PyByteArray_FromStringAndSize((char *) c_uuid, sizeof(uint64_t) * 2);

            //delete (numpy_key);//if deleted, the payload isnt copied because on PyList_SetItem seems to be freed

            PyList_SetItem(py_values, numpy_pos, py_uuid);
            const TupleRow *v;
            try {
                v = parser.make_tuple(py_values, self->T->get_metadata()->get_values());
                self->T->put_crow(k, v);
            }
            catch (std::exception &e) {
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return NULL;
            }
            /*** Done storing the numpy array ***/

            try {
                delete (temp); //Blocking operation
            }
            catch (std::exception &e) {
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return NULL;
            }
        } else {
            //local
            try {
                std::vector<const TupleRow *> value_list = parser.make_tuples_with_npy(py_values, metas);
                for (const TupleRow *T:value_list) {
                    TupleRow *key_copy = new TupleRow(k);
                    self->T->put_crow(key_copy, T);
                }
            }
            catch (std::exception &e) {
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return NULL;
            }
        }
    } else {
        try {
            TupleRow *v = parser.make_tuple(py_values, self->T->get_metadata()->get_values());
            self->T->put_crow(k, v);
            delete(v);
        }
        catch (std::exception &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return NULL;
        }
    }
    delete(k);
    Py_RETURN_NONE;
}

static PyObject *get_row(HCache *self, PyObject *args) {
    PyObject *py_keys, *py_row;
    if (!PyArg_ParseTuple(args, "O", &py_keys)) {
        return NULL;
    }
   const TableMetadata *t_meta = self->T->get_metadata();
    if ((uint32_t)PyList_Size(py_keys)==t_meta->get_keys()->size()) {
        try {
            TupleRow *k = parser.make_tuple(py_keys, self->T->get_metadata()->get_keys());
            std::vector<const TupleRow *> v = self->T->get_crow(k);
            //delete(k); //TODO decide when to do cleanup
            if (self->has_numpy) {
                py_row = parser.merge_blocks_as_nparray(v, self->T->get_metadata()->get_values());
            } else {
                py_row = parser.tuples_as_py(v, self->T->get_metadata()->get_values());
            }
        }
        catch (std::exception &e) {
            std::cerr << e.what() << std::endl;
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return NULL;
        }
    }
    else {
        uint16_t nkeys = (uint16_t) PyList_Size(py_keys);

        const char *table = t_meta->get_table_name();
        const char *keyspace = t_meta->get_keyspace();
        std::shared_ptr<const std::vector<ColumnMeta> > keys_metas = t_meta->get_keys();
        std::shared_ptr<const std::vector<ColumnMeta> > cols_metas = t_meta->get_values();
        std::vector<std::map < std::string, std::string> > keys_conf(nkeys);
        uint32_t ncols = (uint32_t) cols_metas->size()+((uint32_t )keys_metas->size()-nkeys);
        std::vector<std::map < std::string, std::string> > cols_conf(ncols);
        for (uint16_t i = 0; i<nkeys; ++i) {
            keys_conf[i] = keys_metas->at(i).info;
        }
        uint32_t i = 0;
        for (; i<keys_metas->size()-nkeys; ++i) {
            cols_conf[i]=keys_metas->at(i+nkeys).info;

        }
        for (; i<ncols; ++i) {
            cols_conf[i]=cols_metas->at(i-nkeys).info; //TODO I dont like this -nkeys

        }
        std::map<std::string,std::string> config {{"cache_size","0"}};

        CacheTable* randomName = storage->make_cache(table, keyspace, keys_conf, cols_conf, config);
        try {
            TupleRow *k = parser.make_tuple(py_keys, randomName->get_metadata()->get_keys());
            std::vector<const TupleRow *> v = randomName->get_crow(k);
            //delete(k); //TODO decide when to do cleanup
            if (self->has_numpy) {
                py_row = parser.merge_blocks_as_nparray(v, randomName->get_metadata()->get_values());
            } else {
                py_row = parser.tuples_as_py(v, randomName->get_metadata()->get_values());
            }
        }
        catch (std::exception &e) {
            std::cerr << e.what() << std::endl;
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return NULL;
        }
        delete(randomName);

    }
    return py_row;
}


static void hcache_dealloc(HCache *self) {
    delete (self->T);
    self->ob_type->tp_free((PyObject *) self);
}


static PyObject *hcache_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    HCache *self;
    self = (HCache *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}


static int hcache_init(HCache *self, PyObject *args, PyObject *kwds) {
    const char *table, *keyspace, *token_range_pred;

    PyObject *py_tokens, *py_keys_names, *py_cols_names, *py_config;
    if (!PyArg_ParseTuple(args, "sssOOOO", &keyspace, &table, &token_range_pred, &py_tokens,
                          &py_keys_names,
                          &py_cols_names, &py_config)) {
        return -1;
    };


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

    /*** PARSE TABLE METADATA ***/

    uint16_t tokens_size = (uint16_t) PyList_Size(py_tokens);
    uint16_t keys_size = (uint16_t) PyList_Size(py_keys_names);
    uint16_t cols_size = (uint16_t) PyList_Size(py_cols_names);

    self->token_ranges = std::vector<std::pair<int64_t, int64_t>>(tokens_size);
    for (uint16_t i = 0; i < tokens_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_tokens, i);
        int64_t t_a, t_b;
        if (!PyArg_ParseTuple(obj_to_convert, "LL", &t_a, &t_b)) {
            return -1;
        };

        self->token_ranges[i] = std::make_pair(t_a, t_b);
    }


    std::vector<std::map<std::string, std::string> > keys_names(keys_size);

    for (uint16_t i = 0; i < keys_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_keys_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        }
        keys_names[i] = {{"name",std::string(str_temp)}};
    }

    std::vector<std::map<std::string, std::string>> columns_names(cols_size);
    for (uint16_t i = 0; i < cols_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_cols_names, i);
        
        if (PyString_Check(obj_to_convert) || PyUnicode_Check(obj_to_convert)) {
            char *str_temp;
            if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
                return -1;
            };
            columns_names[i] = {{"name",std::string(str_temp)}};
        }
        else if (PyDict_Check(obj_to_convert)) {
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
                config["cache_size"] = "0";
            }
            else columns_names[i]["partition"] = "no-partition";
            self->has_numpy = true;
        }
        else {
            PyErr_SetString(PyExc_RuntimeError, "Can't parse column names, expected String, Dict or Unicode");
            return -1;
        }
    }


    try {
        self->T = storage->make_cache(table, keyspace, keys_names, columns_names, config);
    } catch (ModuleException e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    return 0;
}


static PyMethodDef hcache_type_methods[] = {
        {"get_row",    (PyCFunction) get_row,            METH_VARARGS, NULL},
        {"put_row",    (PyCFunction) put_row,            METH_VARARGS, NULL},
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

    self->token_ranges = std::vector<std::pair<int64_t, int64_t>>(tokens_size);
    for (uint16_t i = 0; i < tokens_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_tokens, i);
        int64_t t_a, t_b;
        if (!PyArg_ParseTuple(obj_to_convert, "LL", &t_a, &t_b)) {
            return -1;
        };

        self->token_ranges[i] = std::make_pair(t_a, t_b);
    }


    std::vector<std::map<std::string, std::string> > keys_names(keys_size);

    for (uint16_t i = 0; i < keys_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_keys_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        }
        keys_names[i] = {{"name",std::string(str_temp)}};
    }

    std::vector<std::map<std::string, std::string>> columns_names(cols_size);
    for (uint16_t i = 0; i < cols_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_cols_names, i);

        if (PyString_Check(obj_to_convert) || PyUnicode_Check(obj_to_convert)) {
            char *str_temp;
            if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
                return -1;
            };
            columns_names[i] = {{"name",std::string(str_temp)}};
        }
        else if (PyDict_Check(obj_to_convert)) {
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
            }
            else columns_names[i]["partition"] = "no-partition";
        }
        else {
            PyErr_SetString(PyExc_RuntimeError, "Can't parse column names, expected String, Dict or Unicode");
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
    } catch (ModuleException e) {
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
    std::shared_ptr<const std::vector<ColumnMeta> > row_metas;
    if (self->P->get_type() == "items") row_metas = self->P->get_metadata()->get_items();
    else if (self->P->get_type() == "values") row_metas = self->P->get_metadata()->get_values();
    else {
        row_metas = self->P->get_metadata()->get_keys();
    }
    PyObject *py_row = parser.tuples_as_py(temp, row_metas);

    if (self->update_cache&&self->P->get_type() != "values") {
        self->baseTable->put_crow(result);
    }
    delete (result);
    return py_row;
}

static void hiter_dealloc(HIterator *self) {
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
        TupleRow *k = parser.make_tuple(py_keys, self->W->get_metadata()->get_keys());
        TupleRow *v = parser.make_tuple(py_values, self->W->get_metadata()->get_values());
        self->W->write_to_cassandra(k, v);
        delete(k);
        delete(v);
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

    std::vector<std::map<std::string, std::string> > keys_names(keys_size);

    for (uint16_t i = 0; i < keys_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_keys_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        }
        keys_names[i] = {{"name",std::string(str_temp)}};
    }

    std::vector<std::map<std::string, std::string>> columns_names(cols_size);
    for (uint16_t i = 0; i < cols_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_cols_names, i);

        if (PyString_Check(obj_to_convert) || PyUnicode_Check(obj_to_convert)) {
            char *str_temp;
            if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
                return -1;
            };
            columns_names[i] = {{"name",std::string(str_temp)}};
        }
        else if (PyDict_Check(obj_to_convert)) {
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
            }
            else columns_names[i]["partition"] = "no-partition";
            self->has_numpy = true;
        }
        else {
            PyErr_SetString(PyExc_RuntimeError, "Can't parse column names, expected String, Dict or Unicode");
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
    } catch (ModuleException e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }

    return 0;
}

static void hwriter_dealloc(HWriter *self) {
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
    }
    else if PyInt_Check((py_config)) {
        int32_t c_val = (int32_t) PyInt_AsLong(py_config);
        config["prefetch_size"]= std::to_string(c_val);
    }
    config["type"] = "items";

    HIterator *iter = (HIterator *) hiter_new(&hfetch_HIterType, args, args);
    iter->baseTable = self->T;
    iter->update_cache = false;
    if (config.find("update_cache") != config.end()) {
        iter->update_cache = true;
    }
    //hiter_init(iter, args, args);
    if (!self->T) {
        PyErr_SetString(PyExc_RuntimeError, "Can't make iterator from null table");
        return NULL;
    }

    try {
        iter->P = storage->get_iterator( self->T->get_metadata(), self->token_ranges, config);
    } catch (ModuleException e) {
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
    }
    else if PyInt_Check((py_config)) {
        int32_t c_val = (int32_t) PyInt_AsLong(py_config);
        config["prefetch_size"]= std::to_string(c_val);
    }
    config["type"] = "keys";

    HIterator *iter = (HIterator *) hiter_new(&hfetch_HIterType, args, args);
    iter->baseTable = self->T;
    //hiter_init(iter, args, args);
    try {
        iter->P = storage->get_iterator(self->T->get_metadata(), self->token_ranges, config);
    } catch (ModuleException e) {
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
    }
    else if PyInt_Check((py_config)) {
        int32_t c_val = (int32_t) PyInt_AsLong(py_config);
        config["prefetch_size"]= std::to_string(c_val);
    }
    config["type"] = "values";

    HIterator *iter = (HIterator *) hiter_new(&hfetch_HIterType, args, args);
    iter->baseTable = self->T;
    //hiter_init(iter, args, args);

    try {
        iter->P = storage->get_iterator(self->T->get_metadata(), self->token_ranges, config);
    } catch (ModuleException e) {
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
    parser = PythonParser();

    PyModule_AddObject(m, "Hcache", (PyObject *) &hfetch_HCacheType);
    PyModule_AddObject(m, "HIterator", (PyObject *) &hfetch_HIterType);
    PyModule_AddObject(m, "HWriter", (PyObject *) &hfetch_HWriterType);
    if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); NUMPY_IMPORT_ARRAY_RETVAL;
    }
}
