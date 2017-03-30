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
        storage = new StorageInterface(nodePort, contact_points);
        parser = new PythonParser();
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
    try {
        TupleRow *k = parser->make_tuple(py_keys, self->metadata->get_keys());
        TupleRow *v = parser->make_tuple(py_values, self->metadata->get_values());
        Py_DecRef(py_keys);
        Py_DecRef(py_values);
        self->T->put_crow(k, v);
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_DecRef(py_keys);
    Py_DecRef(py_values);
    Py_RETURN_NONE;
}

static PyObject *get_row(HCache *self, PyObject *args) {
    PyObject *py_keys, *py_row;
    if (!PyArg_ParseTuple(args, "O", &py_keys)) {
        return NULL;
    }
    try {
        TupleRow *k = parser->make_tuple(py_keys, self->metadata->get_keys());
        Py_DecRef(py_keys);
        const TupleRow *v = self->T->get_crow(k);
        //delete(k); //TODO decide when to do cleanup
        py_row = parser->tuple_as_py(v, self->metadata->get_values());
                Py_INCREF(py_row);
    }
    catch (std::exception &e) {
        std::cout << e.what() << std::endl;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
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


    std::vector<std::string> keys_names = std::vector<std::string>(keys_size);

    for (uint16_t i = 0; i < keys_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_keys_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        }

        keys_names[i] = std::string(str_temp);

    }

    std::vector<std::string> columns_names = std::vector<std::string>(cols_size);
    for (uint16_t i = 0; i < cols_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_cols_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        };
        columns_names[i] = std::string(str_temp);
    }


    /** PARSE CONFIG **/

    std::map<std::string,std::string> config;
    int type_check = PyDict_Check(py_config);
    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return -1;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
          while (PyDict_Next(dict, &pos, &key, &value)){
              std::string conf_key(PyString_AsString(key));
              if (PyString_Check(value)){ std::string conf_val(PyString_AsString(value));
              config[conf_key]=conf_val;}
              if (PyInt_Check(value)){
                  int32_t c_val = (int32_t) PyInt_AsLong(value);
                  config[conf_key]=std::to_string(c_val);}

          }
    }

    try {
        self->T = storage->make_cache(table, keyspace, keys_names, columns_names, config);
        self->metadata=self->T->get_metadata();
      }catch (ModuleException e) {
        PyErr_SetString(PyExc_RuntimeError,e.what());
        return -1;
    }
    return 0;
}


static PyMethodDef hcache_type_methods[] = {
        {"get_row",    (PyCFunction) get_row,            METH_VARARGS, NULL},
        {"put_row",    (PyCFunction) put_row,            METH_VARARGS, NULL},
        {"iterkeys", (PyCFunction) create_iter_keys, METH_VARARGS, NULL},
        {"itervalues", (PyCFunction) create_iter_values, METH_VARARGS, NULL},
        {"iteritems", (PyCFunction) create_iter_items, METH_VARARGS, NULL},
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


    std::vector<std::string> keys_names = std::vector<std::string>(keys_size);

    for (uint16_t i = 0; i < keys_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_keys_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        }
        keys_names[i] = std::string(str_temp);
    }

    std::vector<std::string> columns_names = std::vector<std::string>(cols_size);
    for (uint16_t i = 0; i < cols_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_cols_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        };
        columns_names[i] = std::string(str_temp);
    }


    /** PARSE CONFIG **/

    std::map<std::string,std::string> config;
    int type_check = PyDict_Check(py_config);
    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return -1;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)){
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)){ std::string conf_val(PyString_AsString(value));
                config[conf_key]=conf_val;}
            if (PyInt_Check(value)){
                int32_t c_val = (int32_t) PyInt_AsLong(value);
                config[conf_key]=std::to_string(c_val);}
        }
    }

    try {
        self->P = storage->get_iterator(table,keyspace,keys_names,columns_names,self->token_ranges,config);
        self->metadata=self->P->get_metadata();
    }catch (ModuleException e) {
        PyErr_SetString(PyExc_RuntimeError,e.what());
        return -1;
    }

    return 0;
}


static PyObject *get_next(HIterator *self) {
    const TupleRow* result = self->P->get_cnext();
    if (!result) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    PyObject* py_row = parser->tuple_as_py(result, self->metadata->get_items());

    if (self->update_cache) {
        self->baseTable->put_crow(result);
    }
    else delete(result);
    return py_row;
}

static void hiter_dealloc(HIterator *self) {
    delete (self->P);
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
        TupleRow *k = parser->make_tuple(py_keys, self->metadata->get_keys());
        TupleRow *v = parser->make_tuple(py_values, self->metadata->get_values());
        Py_DecRef(py_keys);
        Py_DecRef(py_values);
        self->W->write_to_cassandra(k,v);
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject *hwriter_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    std::cout << "i try" << std::endl;

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

    std::vector<std::string> keys_names = std::vector<std::string>(keys_size);

    for (uint16_t i = 0; i < keys_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_keys_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        }
        keys_names[i] = std::string(str_temp);
    }

    std::vector<std::string> columns_names = std::vector<std::string>(cols_size);
    for (uint16_t i = 0; i < cols_size; ++i) {
        PyObject *obj_to_convert = PyList_GetItem(py_cols_names, i);
        char *str_temp;
        if (!PyArg_Parse(obj_to_convert, "s", &str_temp)) {
            return -1;
        };
        columns_names[i] = std::string(str_temp);
    }


    /** PARSE CONFIG **/

    std::map<std::string,std::string> config;
    int type_check = PyDict_Check(py_config);
    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return -1;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)){
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)){ std::string conf_val(PyString_AsString(value));
                config[conf_key]=conf_val;}
            if (PyInt_Check(value)){
                int32_t c_val = (int32_t) PyInt_AsLong(value);
                config[conf_key]=std::to_string(c_val);}
        }
    }
    try {
        self->W = storage->make_writer(table,keyspace,keys_names,columns_names,config);
        self->metadata=self->W->get_metadata();
    }catch (ModuleException e) {
        PyErr_SetString(PyExc_RuntimeError,e.what());
        return -1;
    }

    return 0;
}

static void hwriter_dealloc(HWriter *self) {
    delete (self->W);
    self->ob_type->tp_free((PyObject *) self);
}


static PyMethodDef hwriter_type_methods[] = {
        {"write", (PyCFunction) write_cass, METH_VARARGS, NULL},
        {NULL, NULL, 0, NULL}
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
    PyObject* py_config;
    if (!PyArg_ParseTuple(args, "O", &py_config)) {
        return NULL;
    }

    std::map<std::string,std::string> config;
    int type_check = PyDict_Check(py_config);

    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return NULL;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)){
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)){ std::string conf_val(PyString_AsString(value));
                config[conf_key]=conf_val;}
            if (PyInt_Check(value)){
                int32_t c_val = (int32_t) PyInt_AsLong(value);
                config[conf_key]=std::to_string(c_val);}

        }
    }
    config["type"]="items";

    HIterator *iter = (HIterator *) hiter_new(&hfetch_HIterType, args, args);
    iter->baseTable=self->T;
    iter->update_cache = false;
    if (config.find("update_cache")!=config.end()) {
        iter->update_cache=true;
        std::cout << "write cache set" << std::endl;
    }
    //hiter_init(iter, args, args);
    iter->metadata=self->metadata;
    iter->P = storage->get_iterator(self->metadata,self->token_ranges,config);
    return (PyObject *) iter;
}


static PyObject *create_iter_keys(HCache *self, PyObject *args) {
    PyObject* py_config;
    if (!PyArg_ParseTuple(args, "O", &py_config)) {
        return NULL;
    }

    std::map<std::string,std::string> config;
    int type_check = PyDict_Check(py_config);

    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return NULL;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)){
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)){ std::string conf_val(PyString_AsString(value));
                config[conf_key]=conf_val;}
            if (PyInt_Check(value)){
                int32_t c_val = (int32_t) PyInt_AsLong(value);
                config[conf_key]=std::to_string(c_val);}

        }
    }
    config["type"]="keys";

    HIterator *iter = (HIterator *) hiter_new(&hfetch_HIterType, args, args);
    iter->baseTable=self->T;
    //hiter_init(iter, args, args);
    iter->metadata=self->metadata;
    iter->P = storage->get_iterator(self->metadata,self->token_ranges,config);
    return (PyObject *) iter;
}


static PyObject *create_iter_values(HCache *self, PyObject *args) {

    PyObject* py_config;
    if (!PyArg_ParseTuple(args, "O", &py_config)) {
        return NULL;
    }

    std::map<std::string,std::string> config;
    int type_check = PyDict_Check(py_config);

    if (type_check) {
        PyObject *dict;
        if (!PyArg_Parse(py_config, "O", &dict)) {
            return NULL;
        };

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)){
            std::string conf_key(PyString_AsString(key));
            if (PyString_Check(value)){ std::string conf_val(PyString_AsString(value));
                config[conf_key]=conf_val;}
            if (PyInt_Check(value)){
                int32_t c_val = (int32_t) PyInt_AsLong(value);
                config[conf_key]=std::to_string(c_val);}

        }
    }
    config["type"]="values";

    HIterator *iter = (HIterator *) hiter_new(&hfetch_HIterType, args, args);
    iter->baseTable=self->T;
    //hiter_init(iter, args, args);
    iter->metadata=self->metadata;
    iter->P = storage->get_iterator(self->metadata,self->token_ranges,config);
    return (PyObject *) iter;
}

void (*f)(PyObject *) = NULL;

static void module_dealloc(PyObject *self) {
    if (storage) storage->disconnectCassandra();
    //delete(storage);//
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
}