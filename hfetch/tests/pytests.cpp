//
// Created by bscuser on 3/24/17.
//

#include <iostream>
#include <cassandra.h>
#include "gtest/gtest.h"
#include "../CacheTable.h"
#include "../StorageInterface.h"

#include <python2.7/Python.h>

using namespace std;


#define PY_ERR_CHECK if (PyErr_Occurred()){PyErr_Print(); PyErr_Clear();}

const char *keyspace = "test";
const char *particles_table = "particle";
const char *particles_wr_table = "particle_write";
const char *words_wr_table = "words_write";
const char *words_table = "words";
const char *contact_p = "127.0.0.1";

uint32_t nodePort = 9042;

/** TEST SETUP **/

void fireandforget(const char *query, CassSession *session) {
    std::cout << "EXECUTING " << query << std::endl;
    CassStatement *statement = cass_statement_new(query, 0);
    CassFuture *connect_future = cass_session_execute(session, statement);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);
    if (rc != CASS_OK) {
        std::cout << "ERROR ON EXECUTING QUERY: " << cass_error_desc(rc) << std::endl;
    }
    cass_future_free(connect_future);
    cass_statement_free(statement);
}


void setupcassandra() {
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;
    const char *contact_p = "127.0.0.1";
    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect(test_session, test_cluster);
    CassError rc = cass_future_error_code(connect_future);
    cass_future_free(connect_future);

    EXPECT_TRUE(rc == CASS_OK);

    fireandforget("DROP KEYSPACE IF EXISTS test;", test_session);
    fireandforget("CREATE KEYSPACE test WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};",
                  test_session);
    fireandforget(
            "CREATE TABLE test.particle( partid int,time float,x float,y float,z float,ciao text,PRIMARY KEY(partid,time));",
            test_session);
    CassFuture *prepare_future
            = cass_session_prepare(test_session,
                                   "INSERT INTO test.particle(partid , time , x, y , z,ciao ) VALUES (?, ?, ?, ?, ?,?)");
    rc = cass_future_error_code(prepare_future);
    EXPECT_TRUE(rc == CASS_OK);
    const CassPrepared *prepared = cass_future_get_prepared(prepare_future);
    cass_future_free(prepare_future);

    for (int i = 0; i <= 10000; i++) {
        CassStatement *stm = cass_prepared_bind(prepared);
        cass_statement_bind_int32(stm, 0, (cass_int16_t) i);
        cass_statement_bind_float(stm, 1, (cass_float_t) (i / .1));
        cass_statement_bind_float(stm, 2, (cass_float_t) (i / .2));
        cass_statement_bind_float(stm, 3, (cass_float_t) (i / .3));
        cass_statement_bind_float(stm, 4, (cass_float_t) (i / .4));
        cass_statement_bind_string(stm, 5, std::to_string(i * 60).c_str());
        CassFuture *f = cass_session_execute(test_session, stm);
        rc = cass_future_error_code(f);
        EXPECT_TRUE(rc == CASS_OK);
        if (rc != CASS_OK) {
            std::cout << cass_error_desc(rc) << std::endl;
        }
        cass_future_free(f);
        cass_statement_free(stm);
    }
    cass_prepared_free(prepared);

    fireandforget("CREATE TABLE test.words( position int PRIMARY KEY, wordinfo text);", test_session);

    prepare_future = cass_session_prepare(test_session,
                                          "INSERT INTO test.words(position,wordinfo ) VALUES (?, ?)");
    rc = cass_future_error_code(prepare_future);
    EXPECT_TRUE(rc == CASS_OK);

    prepared = cass_future_get_prepared(prepare_future);
    cass_future_free(prepare_future);

    for (int i = 0; i <= 10000; i++) {
        CassStatement *stm = cass_prepared_bind(prepared);
        cass_statement_bind_int32(stm, 0, (cass_int16_t) i);
        std::string val = "IwroteSOMErandomTEXTtoFILLupSPACE" + std::to_string(i * 60);
        cass_statement_bind_string(stm, 1, val.c_str());
        CassFuture *f = cass_session_execute(test_session, stm);
        rc = cass_future_error_code(f);
        EXPECT_TRUE(rc == CASS_OK);
        if (rc != CASS_OK) {
            std::cout << cass_error_desc(rc) << std::endl;
        }
        cass_future_free(f);
        cass_statement_free(stm);
    }


    cass_prepared_free(prepared);
    fireandforget(
            "CREATE TABLE test.particle_write( partid int,time float,x float,y float,z float, PRIMARY KEY(partid,time));",
            test_session);


    fireandforget(
            "CREATE TABLE test.words_write( partid int,time float, x float, ciao text, PRIMARY KEY(partid,time));",
            test_session);

    Py_Initialize();



    _import_array();


    npy_intp dims[2] = {2, 2};
    //void *array = malloc(sizeof(int32_t) * 2);

    double array[4] = {123,456,789,200};

    PyObject *key = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, array);
    PyArrayObject *arr;
    int ok = PyArray_OutputConverter(key, &arr);

    PyObject *bytes = PyArray_ToString(arr, NPY_KEEPORDER);
    PyArray_free(key);

    PyObject* encoded = PyString_AsEncodedObject(bytes, "hex", NULL);
    int a = PyString_GET_SIZE(bytes);
    int b = PyString_GET_SIZE(encoded);

    PyEval_InitThreads();


    prepare_future = cass_session_prepare(test_session,
                                          "INSERT INTO test.bytes(partid,data) VALUES (343, ?)");
    rc = cass_future_error_code(prepare_future);
    EXPECT_TRUE(rc == CASS_OK);
    prepared = cass_future_get_prepared(prepare_future);
    cass_future_free(prepare_future);


    CassStatement *stm = cass_prepared_bind(prepared);
    cass_statement_bind_bytes(stm, 0, reinterpret_cast<const cass_byte_t* >(PyString_AsString(bytes)),a);
    CassFuture *f = cass_session_execute(test_session, stm);
    rc = cass_future_error_code(f);
    EXPECT_TRUE(rc == CASS_OK);
    if (rc != CASS_OK) {
        std::cout << cass_error_desc(rc) << std::endl;
    }
    cass_future_free(f);
    cass_statement_free(stm);

    cass_prepared_free(prepared);
    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "SETTING UP CASSANDRA" << std::endl;
    setupcassandra();
    std::cout << "DONE, CASSANDRA IS UP" << std::endl;

    return RUN_ALL_TESTS();

}



/** PYTHON INTERFACE TESTS **/


#define Py_STRING "s"
#define Py_U_LONGLONG "K"
#define Py_U_LONG "k"
#define Py_LONGLONG "L"
#define Py_LONG "l"
#define Py_BOOL "b"
#define Py_INT "i"
#define Py_U_INT "I"
#define Py_FLOAT "f"
#define Py_DOUBLE "d"
#define Py_SHORT_INT "h"

/** Test to verify Python doubles parsing is performing as expected **/
TEST(TestPyParse, DoubleParse) {
    PyErr_Clear();
    PyObject *key = PyLong_FromDouble(0.12);
    int ok = 0;

    size_t data_size = sizeof(cass_double_t);
    void *data_a = malloc(data_size);
    void *data_b = malloc(data_size);

    cass_double_t t;
    ok = PyArg_Parse(key, Py_DOUBLE, &t);
    EXPECT_EQ(ok, 1);
    memcpy(data_a, &t, sizeof(t));
    ok = PyArg_Parse(key, Py_DOUBLE, data_b);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(memcmp(data_a, data_b, data_size), 0);
    std::free(data_a);
    free(data_b);
    PY_ERR_CHECK
}



TEST(TestingCacheTable, GetRow) {
    PyErr_Clear();
/** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);


/** KEYS **/
    uint32_t key1 = 50;
    float key2 = 500;

    PyObject *list = PyList_New(2);
    PyList_SetItem(list, 0, Py_BuildValue("i", key1));
    PyList_SetItem(list, 1, Py_BuildValue("f", key2));


    float my_float;
    int ok = 0;
    ok = PyArg_Parse(PyList_GetItem(list, 1), "f", &my_float);
    EXPECT_EQ(ok, 1);
    EXPECT_FLOAT_EQ(my_float, key2);

    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::string> colsnames = {"x", "y", "z", "ciao"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};


    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    TableMetadata *table_meta = new TableMetadata(particles_table, keyspace, keysnames, colsnames, test_session);

    CacheTable T = CacheTable(table_meta, test_session, config);


    PyObject *result = T.get_row(list);


    EXPECT_FALSE(result == 0);


    EXPECT_EQ(PyList_Size(result), colsnames.size());
    for (int i = 0; i < PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None == PyList_GetItem(result, i));
    }

    PY_ERR_CHECK

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestingCacheTable, MultiQ) {
    PyErr_Clear();
/** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;


    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect(test_session, test_cluster);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);


    float f1 = 1230;
    PyObject *list = PyList_New(2);
    PyList_SetItem(list, 0, Py_BuildValue("i", 123));
    PyList_SetItem(list, 1, Py_BuildValue("f", f1));
    float f2 = 5430;
    PyObject *list2 = PyList_New(2);
    PyList_SetItem(list2, 0, Py_BuildValue("i", 543));
    PyList_SetItem(list2, 1, Py_BuildValue("f", f2));
    float f3 = 3230;
    PyObject *list3 = PyList_New(2);
    PyList_SetItem(list3, 0, Py_BuildValue("i", 323));
    PyList_SetItem(list3, 1, Py_BuildValue("f", f3));

    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::string> colsnames = {"x", "y", "z", "ciao"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};


    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(
            new CacheTable(particles_table, keyspace, keysnames, colsnames, token_pred, tokens,
                           test_session, config));

    PY_ERR_CHECK
    PyObject *result = T.get()->get_row(list);


    PyObject *result1 = T.get()->get_row(list2);
    Py_DecRef(result1);
    result1 = T.get()->get_row(list3);
    Py_DecRef(result1);
    result1 = T.get()->get_row(list);
    Py_DecRef(result1);
    result1 = T.get()->get_row(list2);
    Py_DecRef(result1);
    result1 = T.get()->get_row(list3);
    Py_DecRef(result1);
    result1 = T.get()->get_row(list);
    Py_DecRef(result1);
    PY_ERR_CHECK

    for (int i = 0; i < PyList_Size(result); ++i) {
        PyList_GetItem(result, i);
    }
    PY_ERR_CHECK

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);


    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestinhMarshallCC, SingleQ) {
    PyErr_Clear();
/** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

/** KEYS **/


    PyObject *list = PyList_New(1);
    PyList_SetItem(list, 0, Py_BuildValue("i", 432));
    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"position"};
    std::vector<std::string> colsnames = {"wordinfo"};
    std::string token_pred = "WHERE token(position)>=? AND token(position)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};


    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(
            new CacheTable(words_table, keyspace, keysnames, colsnames, token_pred, tokens,
                           test_session, config));

    PyObject *result = T.get()->get_row(list);

    PY_ERR_CHECK

    EXPECT_FALSE(result == NULL);


    EXPECT_EQ(PyList_Size(result), colsnames.size());
    for (int i = 0; i < PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None == PyList_GetItem(result, i));
    }

    PY_ERR_CHECK

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestinhMarshall, SingleQ) {
    PyErr_Clear();
/** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

/** SETUP PY **/



    PyObject *list = PyList_New(2);
    int32_t key1 = 645;
    PyList_SetItem(list, 0, Py_BuildValue("i", key1));
    float key2 = 6450;
    PyList_SetItem(list, 1, Py_BuildValue("f", key2));


    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::string> colsnames = {"ciao"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};


    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(
            new CacheTable(particles_table, keyspace, keysnames, colsnames, token_pred, tokens,
                           test_session, config));


    PY_ERR_CHECK
    EXPECT_FLOAT_EQ(PyFloat_AsDouble(PyList_GetItem(list, 1)), key2);


    PY_ERR_CHECK
    PyObject *result = T.get()->get_row(list);

    PY_ERR_CHECK
    EXPECT_FALSE(result == NULL);


    EXPECT_EQ(PyList_Size(result), colsnames.size());
    for (int i = 0; i < PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None == PyList_GetItem(result, i));
    }

    PY_ERR_CHECK

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestingCacheTable, PutFloatsRow) {
/** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

/** SETUP PY **/


/** KEYS **/

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::string> read_colsnames = {"x", "y", "z", "ciao"};
    std::vector<std::string> write_colsnames = {"x", "y", "z"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};


    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    CacheTable ReadTable = CacheTable(particles_table, keyspace, keysnames, read_colsnames, token_pred, tokens,
                                      test_session, config);


    CacheTable WriteTable = CacheTable(particles_wr_table, keyspace, keysnames, write_colsnames, token_pred, tokens,
                                       test_session, config);


    for (int i = 0; i <= 10000; i++) {
        int32_t key1 = i;
        float key2 = (float) (i / .1);
        PY_ERR_CHECK


        PyObject *keys = PyList_New(2);
        PyList_SetItem(keys, 0, Py_BuildValue("i", key1));
        PyList_SetItem(keys, 1, Py_BuildValue("f", key2));

        PyObject *result = ReadTable.get_row(keys);
        PY_ERR_CHECK

        ASSERT_TRUE(result);
        ASSERT_TRUE(PyList_Check(result));
        PyObject *values = PyList_GetSlice(result, keysnames.size() - 2, PyList_Size(result) - 1);
        PY_ERR_CHECK
        WriteTable.put_row(keys, values);
        PY_ERR_CHECK
        Py_DecRef(keys);
        Py_DecRef(values);
        Py_DecRef(result);
    }


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestingCacheTable, PutTextRow) {
/** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

/** KEYS **/

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::string> read_colsnames = {"x", "ciao"};
    std::vector<std::string> write_colsnames = {"x", "ciao"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";

    int64_t bigi = 9223372036854775807;
    std::vector<std::pair<int64_t, int64_t> > tokens = {
            std::pair<int64_t, int64_t>(-bigi - 1, -bigi / 2),
            std::pair<int64_t, int64_t>(-bigi / 2, 0),
            std::pair<int64_t, int64_t>(0, bigi / 2),
            std::pair<int64_t, int64_t>(bigi / 2, bigi)
    };


    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    CacheTable ReadTable = CacheTable(particles_table, keyspace, keysnames, read_colsnames, token_pred, tokens,
                                      test_session, config);


    CacheTable *WriteTable = new CacheTable(words_wr_table, keyspace, keysnames, write_colsnames, token_pred, tokens,
                                            test_session, config);


    Prefetch *P = ReadTable.get_values_iter(100);


    PyObject *result = P->get_next();
    EXPECT_FALSE(result == NULL);
    uint32_t it = 1;
    float fl = 0.03;


    while ((result = P->get_next()) != NULL) {
        PyObject *key = PyList_New(2);
        PyList_SetItem(key, 0, Py_BuildValue("i", it));
        PyList_SetItem(key, 1, Py_BuildValue("f", fl));
        PY_ERR_CHECK
        WriteTable->put_row(key, result);
        PY_ERR_CHECK
        Py_DecRef(key);
        ++it;
    }

    delete (P);
    delete (WriteTable);
    PY_ERR_CHECK
    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}




TEST(TestMetadata,NumpyArrays) {
    ColumnMeta meta = {{"data","double","2x2","no-part"},CASS_VALUE_TYPE_BLOB,0};

    EXPECT_EQ(meta.get_arr_type(),NPY_DOUBLE);
    PyArray_Dims* dims = meta.get_arr_dims();

    EXPECT_EQ(dims->len,2);

    //EXPECT_EQ(*dims->ptr,2);
    //EXPECT_EQ(*(dims->ptr+1),2);
}

/** PYTHON INTERFACE TESTS **/


TEST(TestPythonBlob, TupleRowParsing) {

    /** setup test **/
    PyErr_Clear();

    _import_array();
    npy_intp dims[2] = {2, 2};
    void *array = malloc(sizeof(double) * 4);

    double *temp = (double *) array;
    *temp = 123;
    *(temp + 1) = 456;
    *(temp + 2) = 789;
    *(temp + 3) = 500;
    PyObject *key = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, array);

    /** interface receives key **/

    PyArrayObject *arr;
    int ok = PyArray_OutputConverter(key, &arr);
    if (!ok) throw ModuleException("error parsing PyArray to obj");

    /** transform to bytes **/
    PyObject *bytes = PyArray_ToString(arr, NPY_KEEPORDER);
    PY_ERR_CHECK

    ok = PyString_Check(bytes);
    PY_ERR_CHECK
    Py_ssize_t l_size = PyString_Size(bytes);
    PY_ERR_CHECK

    // store bytes
    void *data = malloc(l_size);
    char *l_temp = PyString_AsString(bytes);
    PY_ERR_CHECK
    char *permanent = (char *) malloc(l_size + 1);
    memcpy(permanent, l_temp, l_size);
    permanent[l_size] = '\0';
    memcpy(data, &permanent, sizeof(char *));
    PY_ERR_CHECK

    /** cleanup **/
    Py_DecRef(key);

    free(data);
    free(permanent);
}


/** Test to verify Python doubles parsing is performing as expected **/
TEST(TestPyParse, DoubleParse) {
    PyErr_Clear();
    PyObject *key = PyLong_FromDouble(0.12);
    int ok = 0;

    size_t data_size = sizeof(cass_double_t);
    void *data_a = malloc(data_size);
    void *data_b = malloc(data_size);

    cass_double_t t;
    ok = PyArg_Parse(key, Py_DOUBLE, &t);
    EXPECT_EQ(ok, 1);
    memcpy(data_a, &t, sizeof(t));
    ok = PyArg_Parse(key, Py_DOUBLE, data_b);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(memcmp(data_a, data_b, data_size), 0);
    std::free(data_a);
    free(data_b);
    PY_ERR_CHECK
}


TEST(TestingCacheTable, GetRow) {
    PyErr_Clear();
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);


    /** KEYS **/
    uint32_t key1 = 50;
    float key2 = 500;

    PyObject *list = PyList_New(2);
    PyList_SetItem(list, 0, Py_BuildValue("i", key1));
    PyList_SetItem(list, 1, Py_BuildValue("f", key2));


    float my_float;
    int ok = 0;
    ok = PyArg_Parse(PyList_GetItem(list, 1), "f", &my_float);
    EXPECT_EQ(ok, 1);
    EXPECT_FLOAT_EQ(my_float, key2);

    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::vector<std::string> > colsnames = {std::vector<std::string>{"x"}, std::vector<std::string>{"y"},
                                                        std::vector<std::string>{"z"},
                                                        std::vector<std::string>{"ciao"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};


    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";
    CacheTable T = CacheTable(particles_table, keyspace, keysnames, colsnames, token_pred, tokens,
                              test_session, config);



    PyObject *result = T.get_row(list);


    EXPECT_FALSE(result == 0);


    EXPECT_EQ(PyList_Size(result), colsnames.size());
    for (int i = 0; i < PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None == PyList_GetItem(result, i));
    }

    PY_ERR_CHECK

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestingCacheTable, NumpyArrayWriteBig) {
    PyErr_Clear();

    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);


    uint32_t key1 = 343;
    PyObject *list = PyList_New(1);
    PyList_SetItem(list, 0, Py_BuildValue("i", key1));


    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"partid"};
    std::vector<std::vector<std::string> > colsnames = {std::vector<std::string>{"image","double","2x2","partition","arrays_aux"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";



    CacheTable *T = new CacheTable("arrays", keyspace,keysnames, colsnames, token_pred, tokens,test_session,config);


    _import_array();


    npy_intp dims[2] = {2, 2};
    void *array = malloc(sizeof(double) * 4);

    double *temp = (double *) array;
    *temp = 123;
    *(temp + 1) = 456;
    *(temp + 2) = 789;
    *(temp + 3) = 200;

    PyObject *py_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, array);

    PyObject* rw_list = PyList_New(1);
    PyList_SetItem(rw_list,0,py_array);
    T->put_row(list,rw_list);

    delete(T);

    free(array);


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

    PY_ERR_CHECK

}



TEST(TestingCacheTable, NumpyArrayReadWriteBig) {
    PyErr_Clear();

    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);


    uint32_t key1 = 343;
    PyObject *list = PyList_New(1);
    PyList_SetItem(list, 0, Py_BuildValue("i", key1));


    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"partid"};
    std::vector<std::vector<std::string> > colsnames = {std::vector<std::string>{"image","double","2x2","partition","arrays_aux"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";



    CacheTable *T = new CacheTable("arrays", keyspace,keysnames, colsnames, token_pred, tokens,test_session,config);


    _import_array();


    npy_intp dims[2] = {2, 2};
    void *array = malloc(sizeof(double) * 4);

    double *temp = (double *) array;
    *temp = 123;
    *(temp + 1) = 456;
    *(temp + 2) = 789;
    *(temp + 3) = 200;

    PyObject *py_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, array);

    PyObject* rw_list = PyList_New(1);
    PyList_SetItem(rw_list,0,py_array);
    T->put_row(list,rw_list);

    for (int i = 0; i<100000; ++i) {
        double d = 12.34;
        d=d*(i-d+13);
    }
    PyObject* result = T->get_row(list);
    EXPECT_TRUE(result!=NULL);
    delete(T);

    free(array);


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

    PY_ERR_CHECK

}




TEST(TestingCacheTable, NumpyArrayReadWrite) {
    PyErr_Clear();
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);


    /** KEYS **/
    uint32_t key1 = 343;
    PyObject *list = PyList_New(1);
    PyList_SetItem(list, 0, Py_BuildValue("i", key1));


    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"partid"};
    std::vector<std::vector<std::string> > colsnames = {std::vector<std::string>{"data","double","2x2","no-part"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";



    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(new CacheTable(bytes_table, keyspace,
                                                                               keysnames, colsnames, token_pred, tokens,
                                                                               test_session,config));

    PyObject *result = T.get()->get_row(list);

    EXPECT_FALSE(result == 0);

    EXPECT_EQ(PyList_Size(result), colsnames.size());
    for (int i = 0; i < PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None == PyList_GetItem(result, i));
    }
    _import_array();
    int check = PyArray_Check(PyList_GetItem(result, 0));
    EXPECT_TRUE(check);

    PyArrayObject *rewrite, *rewrite2;
    PyArray_OutputConverter(PyList_GetItem(result,0),&rewrite);
    PyObject* rewr_obj = PyArray_Transpose(rewrite,NULL);
    check = PyArray_Check(rewr_obj);
    EXPECT_TRUE(check);
    PyObject* rw_list = PyList_New(1);
    PyList_SetItem(rw_list,0,rewr_obj);
    T->put_row(list,rw_list);

    PY_ERR_CHECK





    result = T.get()->get_row(list);
    EXPECT_FALSE(result == 0);
    ASSERT_TRUE(PyList_Check(result));
    ASSERT_GT(PyList_Size(result),0);
    check = PyArray_Check(PyList_GetItem(result, 0));
    EXPECT_TRUE(check);



    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

    PY_ERR_CHECK

}



TEST(TestingCacheTable, NumpyArrayRead) {
    PyErr_Clear();

    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);


    /** KEYS **/
    uint32_t key1 = 343;
    PyObject *list = PyList_New(1);
    PyList_SetItem(list, 0, Py_BuildValue("i", key1));


    PY_ERR_CHECK

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    std::vector<std::string> keysnames = {"partid"};
    std::vector<std::vector<std::string> > colsnames = {std::vector<std::string>{"data","double","2x2","no-part"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};
    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(new CacheTable(bytes_table, keyspace,
                                                                               keysnames, colsnames, token_pred, tokens,
                                                                               test_session,config));


    PyObject *result = T.get()->get_row(list);


    EXPECT_FALSE(result == 0);


    EXPECT_EQ(PyList_Size(result), colsnames.size());
    for (int i = 0; i < PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None == PyList_GetItem(result, i));
    }

    int check = PyArray_Check(PyList_GetItem(result, 0));
    EXPECT_TRUE(check);
    PY_ERR_CHECK

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

    PY_ERR_CHECK

}



TEST(TestingCacheTable, MultiQ) {
    PyErr_Clear();
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;


    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect(test_session, test_cluster);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);


    float f1 = 1230;
    PyObject *list = PyList_New(2);
    PyList_SetItem(list, 0, Py_BuildValue("i", 123));
    PyList_SetItem(list, 1, Py_BuildValue("f", f1));
    float f2 = 5430;
    PyObject *list2 = PyList_New(2);
    PyList_SetItem(list2, 0, Py_BuildValue("i", 543));
    PyList_SetItem(list2, 1, Py_BuildValue("f", f2));
    float f3 = 3230;
    PyObject *list3 = PyList_New(2);
    PyList_SetItem(list3, 0, Py_BuildValue("i", 323));
    PyList_SetItem(list3, 1, Py_BuildValue("f", f3));

    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::vector<std::string> > colsnames = {std::vector<std::string>{"x"}, std::vector<std::string>{"y"},
                                                        std::vector<std::string>{"z"},
                                                        std::vector<std::string>{"ciao"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};



    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(
            new  CacheTable(particles_table, keyspace, keysnames, colsnames, token_pred, tokens,
                            test_session, config));

    PY_ERR_CHECK
    PyObject *result = T.get()->get_row(list);


    PyObject *result1 = T.get()->get_row(list2);
    PyObject *result2 = T.get()->get_row(list3);
    PyObject *result3 = T.get()->get_row(list);
    PyObject *result4 = T.get()->get_row(list2);
    PyObject *result5 = T.get()->get_row(list3);
    PyObject *result6 = T.get()->get_row(list);
    PY_ERR_CHECK

    for (int i = 0; i < PyList_Size(result); ++i) {
        PyList_GetItem(result, i);
    }
    PY_ERR_CHECK

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);


    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestinhMarshallCC, SingleQ) {
    PyErr_Clear();
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

    /** KEYS **/


    PyObject *list = PyList_New(1);
    PyList_SetItem(list, 0, Py_BuildValue("i", 432));
    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"position"};
    std::vector<std::vector<std::string> > colsnames = {std::vector<std::string>{"wordinfo"}};
    std::string token_pred = "WHERE token(position)>=? AND token(position)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};




    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(
            new  CacheTable(words_table, keyspace, keysnames, colsnames, token_pred, tokens,
                            test_session, config));

    PyObject *result = T.get()->get_row(list);

    PY_ERR_CHECK

    EXPECT_FALSE(result == NULL);


    EXPECT_EQ(PyList_Size(result), colsnames.size());
    for (int i = 0; i < PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None == PyList_GetItem(result, i));
    }

    PY_ERR_CHECK

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestinhMarshall, SingleQ) {
    PyErr_Clear();
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

    /** SETUP PY **/



    PyObject *list = PyList_New(2);
    int32_t key1 = 645;
    PyList_SetItem(list, 0, Py_BuildValue("i", key1));
    float key2 = 6450;
    PyList_SetItem(list, 1, Py_BuildValue("f", key2));


    PY_ERR_CHECK

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::vector<std::string> > colsnames = {std::vector<std::string>{"ciao"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};





    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(
            new  CacheTable(particles_table, keyspace, keysnames, colsnames, token_pred, tokens,
                            test_session, config));


    PY_ERR_CHECK
    EXPECT_FLOAT_EQ(PyFloat_AsDouble(PyList_GetItem(list,1)),key2);


    PY_ERR_CHECK
    PyObject *result = T.get()->get_row(list);

    PY_ERR_CHECK
    EXPECT_FALSE(result == NULL);


    EXPECT_EQ(PyList_Size(result), colsnames.size());
    for (int i = 0; i < PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None == PyList_GetItem(result, i));
    }

    PY_ERR_CHECK

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestingCacheTable, PutFloatsRow) {
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

    /** SETUP PY **/


    /** KEYS **/

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::vector<std::string> > read_colsnames = {std::vector<std::string>{"x"},
                                                             std::vector<std::string>{"y"},
                                                             std::vector<std::string>{"z"},
                                                             std::vector<std::string>{"ciao"}};
    std::vector<std::vector<std::string> > write_colsnames = {std::vector<std::string>{"x"},
                                                              std::vector<std::string>{"y"},
                                                              std::vector<std::string>{"z"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};


    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";



    CacheTable ReadTable = CacheTable(particles_table, keyspace, keysnames, read_colsnames, token_pred, tokens,
                                      test_session, config);


    CacheTable WriteTable = CacheTable(particles_wr_table, keyspace, keysnames, write_colsnames, token_pred, tokens,
                                       test_session, config);


    for (int i = 0; i <= 10000; i++) {
        int32_t key1 = i;
        float key2 = (float) (i / .1);
        PY_ERR_CHECK


        PyObject *keys = PyList_New(2);
        PyList_SetItem(keys, 0, Py_BuildValue("i", key1));
        PyList_SetItem(keys, 1, Py_BuildValue("f", key2));

        PyObject *result = ReadTable.get_row(keys);
        PY_ERR_CHECK

        ASSERT_TRUE(result);
        ASSERT_TRUE(PyList_Check(result));
        PyObject *values = PyList_GetSlice(result, keysnames.size() - 2, PyList_Size(result) - 1);
        PY_ERR_CHECK
        WriteTable.put_row(keys, values);
        PY_ERR_CHECK
        Py_DecRef(keys);
        Py_DecRef(values);
        Py_DecRef(result);
    }


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestingCacheTable, PutTextRow) {
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

    /** KEYS **/

    std::vector<std::string> keysnames = {"partid", "time"};

    std::vector<std::vector<std::string> > read_colsnames = {std::vector<std::string>{"x"},
                                                             std::vector<std::string>{"ciao"}};
    std::vector<std::vector<std::string> > write_colsnames = {std::vector<std::string>{"x"},
                                                              std::vector<std::string>{"ciao"}};

    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";

    int64_t bigi= 9223372036854775807;
    std::vector<std::pair<int64_t, int64_t> > tokens = {
            std::pair<int64_t, int64_t>(-bigi -1,  -bigi/2),
            std::pair<int64_t, int64_t>(-bigi/2,0),
            std::pair<int64_t, int64_t>(0,bigi/2),
            std::pair<int64_t, int64_t>(bigi/2, bigi)
    };


    <<<<<<<<< Temporary merge branch 1
    CacheTable ReadTable = CacheTable(max_items, particles_table, keyspace, keysnames, read_colsnames, token_pred,
                                      tokens,
                                      test_session);


    CacheTable *WriteTable = new CacheTable(max_items, words_wr_table, keyspace, keysnames, write_colsnames, token_pred,
                                            tokens,
                                            test_session);
    =========

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    CacheTable ReadTable = CacheTable(particles_table, keyspace, keysnames, read_colsnames, token_pred, tokens,
                                      test_session, config);


    CacheTable *WriteTable = new CacheTable(words_wr_table, keyspace, keysnames, write_colsnames, token_pred, tokens,
                                            test_session, config);



    Prefetch *P = ReadTable.get_values_iter(100);


    PyObject *result = P->get_next();
    EXPECT_FALSE(result == NULL);
    uint32_t it = 1;
    float fl = 0.03;

    PyObject *key;
    while (result != NULL) {
        PY_ERR_CHECK
        key = PyList_New(0);
        PY_ERR_CHECK
        PyList_Append(key, Py_BuildValue("i", it));
        PyList_Append(key,Py_BuildValue("f", fl));
        PY_ERR_CHECK
        WriteTable->put_row(key, result);
        PY_ERR_CHECK
        Py_DECREF(key);//Py_DecRef(key);
        PY_ERR_CHECK
        ++it;
        result = P->get_next();
    }

//    WriteTable.flush_elements(); //Blocking OP
    //Handle correctly who destroys the data inserted in cassandra



    delete (WriteTable);
    delete (P);
    PY_ERR_CHECK
    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}