 #include "HCache.h"


typedef struct {
    PyObject_HEAD
    CacheTable *T;
    Prefetch *P;
} HCache;



static PyObject *connectCassandra(PyObject * self,PyObject *args) {
    int nodePort;
    char *contact_points;

    int ok=  PyArg_ParseTuple(args, "is", &nodePort, &contact_points);
    assert(ok);

    CassFuture *connect_future = NULL;
    cluster = cass_cluster_new();
    session = cass_session_new();
    // add contact points
    cass_cluster_set_contact_points(cluster, contact_points);
    cass_cluster_set_port(cluster, nodePort);
    // Provide the cluster object as configuration to connect the session
    connect_future = cass_session_connect(session, cluster);
    CassError rc = cass_future_error_code(connect_future);
    if (rc != CASS_OK) {
        throw std::runtime_error(cass_error_desc(rc));
    }
    cass_future_free(connect_future);
    Py_RETURN_TRUE;
}


static PyObject *disconnectCassandra(PyObject * self) {

/* IF SETUP SUCCESSFUL */
    if (session != NULL) {
        CassFuture *close_future = cass_session_close(session);
        cass_future_wait(close_future);
        cass_future_free(close_future);
    }

/* ALWAYS */
    cass_cluster_free(cluster);
    cass_session_free(session);
    Py_RETURN_TRUE;
}



/*** MODULE SETUP ***/




static PyObject* put_row(HCache * self, PyObject *args) {
    PyObject *py_row;
    int ok = PyArg_ParseTuple(args, "O", &py_row);
    assert(ok);
    int success = self->T->put_row(py_row);
     assert(success);
     Py_RETURN_NONE;
}


 static PyObject *get_row(HCache * self, PyObject *args) {
     PyObject *py_keys;
     int ok= PyArg_ParseTuple(args, "O", &py_keys);
     assert(ok);
     return  self->T->get_row(py_keys);
 }



 static PyObject *get_next(HCache * self) {
     return  self->P->get_next();
 }


 static void
Cache_dealloc(HCache* self)
{
    delete(self->T);
    delete(self->P);
    self->ob_type->tp_free((PyObject*)self);
}

 static PyObject* Cache_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
 {
     HCache *self;
     self = (HCache *)type->tp_alloc(type, 0);
     return (PyObject *) self;
 }

 static int Cache_init(HCache *self, PyObject *args, PyObject *kwds)
 {
     const char *table, *keyspace, *query_get,*query_token;
     uint32_t cache_size, prefetch_size;
     PyObject *py_tokens;
     int ok = PyArg_ParseTuple(args, "IsssOIs", &cache_size, &table, &keyspace, &query_get,&py_tokens,&prefetch_size,&query_token);
     assert(ok);

     std::pair<uint64_t,uint64_t> *token_ranges = new std::pair<uint64_t ,uint64_t >[PyList_Size(py_tokens)];
     for(uint16_t i =0; i<PyList_Size(py_tokens); ++i){
         PyObject* obj_to_convert=PyList_GetItem(py_tokens,i);
         uint64_t t_a,t_b;
         ok = PyArg_ParseTuple(obj_to_convert,"KK",&t_a,&t_b);
         token_ranges[i]=std::make_pair(t_a,t_b);
     }
     self->T = new CacheTable((uint32_t) cache_size, table, keyspace, query_get, session);
     self->P = new Prefetch(token_ranges,prefetch_size,self->T,session,query_token,PyList_Size(py_tokens));
     return 0;
 }


static PyMethodDef type_methods[] = {
        {"get_row",             (PyCFunction) get_row,             METH_VARARGS, NULL},
        {"put_row",             (PyCFunction) put_row,             METH_VARARGS, NULL},
        {"get_next",             (PyCFunction) get_next,             METH_VARARGS, NULL},
        {NULL, NULL, 0,                                                          NULL}
};

static PyMethodDef module_methods[] = {
        {"connectCassandra",    (PyCFunction) connectCassandra,    METH_VARARGS, NULL},
        {"disconnectCassandra", (PyCFunction) disconnectCassandra, METH_NOARGS,  NULL},
        {NULL, NULL, 0,                                                          NULL}
};


static PyTypeObject hfetch_CacheType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "hfetch.cache",             /* tp_name */
        sizeof(HCache), /* tp_basicsize */
        0,                         /*tp_itemsize*/
        (destructor)Cache_dealloc, /*tp_dealloc*/
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
        0,		               /* tp_traverse */
        0,		               /* tp_clear */
        0,		               /* tp_richcompare */
        0,		               /* tp_weaklistoffset */
        0,		               /* tp_iter */
        0,		               /* tp_iternext */
        type_methods,             /* tp_methods */
        0,             /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)Cache_init,      /* tp_init */
        0,                         /* tp_alloc */
        Cache_new,                 /* tp_new */
};


PyMODINIT_FUNC
inithfetch(void)
{
    PyObject* m;
    hfetch_CacheType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&hfetch_CacheType) < 0)
        return;

    m = Py_InitModule3("hfetch", module_methods, "docstring...");

    Py_INCREF(&hfetch_CacheType);
    PyModule_AddObject(m, "hcache", (PyObject *)&hfetch_CacheType);
}