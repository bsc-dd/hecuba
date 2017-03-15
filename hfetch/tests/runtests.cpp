#include <iostream>
#include <cassandra.h>
#include "gtest/gtest.h"
#include "../CacheTable.h"


using namespace std;


#define PY_ERR_CHECK if (PyErr_Occurred()){PyErr_Print(); PyErr_Clear();}

const char *keyspace = "test";
const char *particles_table = "particle";
const char *particles_wr_table = "particle_write";
const char *words_wr_table = "words_write";
const char *words_table = "words";
const char *contact_p = "127.0.0.1";

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
    cass_cluster_set_port(test_cluster, 9042);

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
    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

    Py_Initialize();

    PyEval_InitThreads();

}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "SETTING UP CASSANDRA" << std::endl;
    setupcassandra();
    std::cout << "DONE, CASSANDRA IS UP" << std::endl;

    return RUN_ALL_TESTS();

}


/** Test to verify Intel's TBB is performing as expected **/
TEST(TestTBBQueue, Try_pop) {
    EXPECT_TRUE(TBB_USE_EXCEPTIONS);
    PyObject *key = PyInt_FromSize_t(123);
    PyObject *value = PyInt_FromSize_t(233);
    tbb::concurrent_bounded_queue<std::pair<PyObject *, PyObject *> > data;
    data.push(std::make_pair(key, value));

    std::pair<PyObject *, PyObject *> item;
    EXPECT_TRUE(data.try_pop(item));
    EXPECT_EQ(PyInt_AsLong(item.first), 123);
    EXPECT_EQ(PyInt_AsLong(item.second), 233);
}


/** Test to asses Poco Cache is performing as expected with pointer **/
TEST(TestingPocoCache, InsertGetDeleteOps) {
    const uint16_t i = 123;
    const uint16_t j = 456;
    size_t ss = sizeof(uint16_t) * 2;
    Poco::LRUCache<TupleRow, TupleRow> myCache(2);

    ColumnMeta cm1={0,CASS_VALUE_TYPE_INT,"ciao"};
    ColumnMeta cm2 ={sizeof(uint16_t),CASS_VALUE_TYPE_INT,"ciaociao"};
    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);



    char *b2 = (char *) malloc(ss);
    memcpy(b2, &i, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &j, sizeof(uint16_t));
    const TupleRow *t1 = new TupleRow(metas,sizeof(uint16_t)*2,b2);

    uint16_t ka = 64;
    uint16_t kb = 128;
    b2 = (char *) malloc(ss);
    memcpy(b2, &ka, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &kb, sizeof(uint16_t));


    TupleRow *key1 = new TupleRow( metas,sizeof(uint16_t) * 2, b2);
    myCache.add(*key1, t1);

    EXPECT_EQ(myCache.getAllKeys().size(), 1);
    TupleRow t = *(myCache.getAllKeys().begin());
    EXPECT_TRUE(t == *key1);
    EXPECT_FALSE(&t == key1);
    /**
     * Reason: Cache builds its own copy of key1 through the copy constructor. They are equal but not the same object
     **/
    myCache.clear();
    //Removes all references, and deletes all objects. Key1 is still active thanks to our ref
    delete (key1);
}



TEST(TestingPocoCache, ReplaceOp) {
    uint16_t i = 123;
    uint16_t j = 456;
    size_t ss = sizeof(uint16_t) * 2;
    Poco::LRUCache<TupleRow, TupleRow> myCache(2);

    ColumnMeta cm1={0,CASS_VALUE_TYPE_INT,"ciao"};
    ColumnMeta cm2 ={sizeof(uint16_t),CASS_VALUE_TYPE_INT,"ciaociao"};
    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);



    char *b2 = (char *) malloc(ss);
    memcpy(b2, &i, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &j, sizeof(uint16_t));
    const TupleRow *t1 = new TupleRow(metas,sizeof(uint16_t)*2,b2);

    uint16_t ka = 64;
    uint16_t kb = 128;
    b2 = (char *) malloc(ss);
    memcpy(b2, &ka, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &kb, sizeof(uint16_t));


    TupleRow *key1 = new TupleRow( metas,sizeof(uint16_t) * 2, b2);
    myCache.add(key1, t1);

    EXPECT_EQ(myCache.getAllKeys().size(), 1);
    TupleRow t = *(myCache.getAllKeys().begin());
    EXPECT_TRUE(t == *key1);
    EXPECT_FALSE(&t == key1);


    delete(t1);

    i=500;
    b2 = (char *) malloc(ss);
    memcpy(b2, &i, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &j, sizeof(uint16_t));
    const TupleRow *t2 = new TupleRow(metas,sizeof(uint16_t)*2,b2);
    myCache.add(*key1,t2);
    i=123;
    b2 = (char *) malloc(ss);
    memcpy(b2, &i, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &j, sizeof(uint16_t));
    const TupleRow *t3 = new TupleRow(metas,sizeof(uint16_t)*2,b2);
    myCache.add(key1,t3);

    EXPECT_EQ(myCache.getAllKeys().size(), 1);
    t = *(myCache.getAllKeys().begin());
    EXPECT_TRUE(t == *key1);
    EXPECT_FALSE(&t == key1);
    t = *(myCache.get(t));
    EXPECT_TRUE(t == *t3);
    EXPECT_FALSE(&t == t3);

    /**
     * Reason: Cache builds its own copy of key1 through the copy constructor. They are equal but not the same object
     **/
    myCache.clear();
    //Removes all references, and deletes all objects. Key1 is still active thanks to our ref
    delete (key1);
}





/** Testing custom comparators for TupleRow **/
TEST(TupleTest, TupleOps) {
    const uint16_t i = 123;
    const uint16_t j = 456;
    size_t size = sizeof(uint16_t);
    auto sizes = vector<uint16_t>(2, size);
    char *buffer = (char *) malloc(size * 2);
    char *buffer2 = (char *) malloc(size * 2);
    memcpy(buffer, &i, size);
    memcpy(buffer + size, &j, size);
    memcpy(buffer2, &i, size);
    memcpy(buffer2 + size, &j, size);

    ColumnMeta cm1={0,CASS_VALUE_TYPE_INT,"ciao"};
    ColumnMeta cm2 ={sizeof(uint16_t),CASS_VALUE_TYPE_INT,"ciaociao"};
    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);



    TupleRow t1 = TupleRow(metas,sizeof(uint16_t) * 2, buffer);
    TupleRow t2 = TupleRow(metas,sizeof(uint16_t) * 2, buffer2);

    //Equality
    EXPECT_TRUE(!(t1 < t2) && !(t2 < t1));
    EXPECT_TRUE(!(t1 > t2) && !(t2 > t1));

    cm2 ={sizeof(uint16_t),CASS_VALUE_TYPE_INT,"ciaocia"};
    std::vector<ColumnMeta> v2 = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas2=std::make_shared<std::vector<ColumnMeta>>(v2);


    char *buffer3 = (char *) malloc(size * 2);
    memcpy(buffer3, &i, size);
    memcpy(buffer3 + size, &j, size);
    //Though their inner Metadata has the same values, they are different objects
    TupleRow t3 = TupleRow(metas2,sizeof(uint16_t) * 2, buffer3);
    EXPECT_FALSE(!(t1 < t3) && !(t3 < t1));
    EXPECT_FALSE(!(t1 > t3) && !(t3 > t1));

}



/** PURE C++ TESTS **/
TEST(TestingCacheTable, GetRowC) {
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

    char *buffer = (char *) malloc(sizeof(int) + sizeof(float));

    int val = 1234;
    memcpy(buffer, &val, sizeof(int));

    float f = 12340;
    memcpy(buffer + sizeof(int), &f, sizeof(float));


    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::string> colsnames = {"x", "y", "z", "ciao"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};
    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";
    CacheTable *CTable = new CacheTable(particles_table, keyspace, keysnames, colsnames, token_pred, tokens,
                              test_session, config);
    TupleRow *t = new TupleRow(CTable->_test_get_keys_factory()->get_metadata(), sizeof(int) + sizeof(float), buffer);


    const TupleRow *result = CTable->get_crow(t);

    EXPECT_FALSE(result == NULL);

    if (result != 0) {
        float *p = (float *) result->get_element(0);
        EXPECT_FLOAT_EQ((float) (*p), 6170);

        p = (float *) result->get_element(1);
        EXPECT_FLOAT_EQ((float) (*p), 4113.3335);

        p = (float *) result->get_element(2);
        EXPECT_FLOAT_EQ((float) (*p), 3085);

        // const void *v=result->get_element(3);
        int64_t *addr = (int64_t *) result->get_element(3);
        //memcpy(&addr,v,sizeof(char*));
        char *d = reinterpret_cast<char *>(*addr);

        EXPECT_STREQ(d, "74040");

    }

    delete(t);
    delete(result);
    delete(CTable);


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

}


TEST(TestingCacheTable, GetRowStringC) {
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

    char *buffer = (char *) malloc(sizeof(int) + sizeof(float));

    int val = 1234;
    memcpy(buffer, &val, sizeof(int));

    float f = 12340;
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    std::vector<uint16_t> offsets = {0, sizeof(int)};


    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::string> colsnames = {"ciao"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";
    CacheTable T = CacheTable(particles_table, keyspace, keysnames, colsnames, token_pred, tokens,
                                        test_session, config);


    TupleRow *t = new TupleRow(T._test_get_keys_factory()->get_metadata(), sizeof(int) + sizeof(float), buffer);

    const TupleRow *result = T.get_crow(t);

    EXPECT_FALSE(result == NULL);

    if (result != 0) {

        const void *v = result->get_element(0);
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "74040");

    }

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestingCacheTable, PutRowStringC) {

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

    char *buffer = (char *) malloc(sizeof(int) + sizeof(float));

    int val = 1234;
    memcpy(buffer, &val, sizeof(int));

    float f = 12340;
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    std::vector<uint16_t> offsets = {0, sizeof(int)};


    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::string> colsnames = {"ciao"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";



    CacheTable T = CacheTable(particles_table, keyspace, keysnames, colsnames, token_pred, tokens,
                              test_session,config);



    //TupleRow *t = new TupleRow(T._test_get_keys_factory()->get_metadata(), sizeof(int) + sizeof(float), buffer);

    std::shared_ptr<void> result = T.get_crow(buffer);

    EXPECT_FALSE(result == NULL);

    if (result != NULL) {

        const void *v = result.get();
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "74040");
        const char* substitue = "71919";
        EXPECT_EQ(std::strlen(d),std::strlen(substitue));

        memcpy(d, substitue, std::strlen(d));
    }


     T.put_crow(buffer,result.get());


    result = T.get_crow(buffer);


    EXPECT_FALSE(result == NULL);
/*
    if (result != 0) {
        const void *v = result.get();
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "71919");
        const char* substitue = "74040";
        EXPECT_EQ(std::strlen(d),std::strlen(substitue));


        memcpy(d, substitue, std::strlen(d));
    }
*/
    result = T.get_crow(buffer);
   // T.put_crow(buffer,result.get());

    result = T.get_crow(buffer);
  /*  if (result != 0) {

        const void *v = result.get();
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "74040");
    }
*/
    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestingPrefetch, GetNextC) {
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

    char *buffer = (char *) malloc(sizeof(int) + sizeof(float));

    int val = 1234;
    memcpy(buffer, &val, sizeof(int));

    float f = 12340;
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    ColumnMeta cm1={0,CASS_VALUE_TYPE_INT,"partid"};
    ColumnMeta cm2 ={sizeof(uint16_t),CASS_VALUE_TYPE_FLOAT,"time"};
    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);

    TupleRow *t = new TupleRow(metas, sizeof(int) + sizeof(float), buffer);

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<std::string> colsnames = {"ciao"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    int64_t bigi= 9223372036854775807;
    std::vector<std::pair<int64_t, int64_t> > tokens = {
            std::pair<int64_t, int64_t>(-bigi -1,  -bigi/2),
            std::pair<int64_t, int64_t>(-bigi/2,0),
            std::pair<int64_t, int64_t>(0,bigi/2),
            std::pair<int64_t, int64_t>(bigi/2, bigi)
    };


    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";
    CacheTable T = CacheTable(particles_table, keyspace, keysnames, colsnames, token_pred, tokens,
                              test_session, config);


    Prefetch *P = T.get_values_iter(100);


    TupleRow *result = P->get_cnext();
    EXPECT_FALSE(result == NULL);
    uint16_t it = 1;


    while ((result = P->get_cnext())!=NULL) {
        const void *v = result->get_element(0);
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);
        std::string empty_str="";
        std::string result_str(d);
        EXPECT_TRUE(result_str>empty_str);
        ++it;
    }

    EXPECT_EQ(it,10001);

    delete(P);

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}




/** PYTHON INTERFACE TESTS **/

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
    std::vector<std::string> colsnames = {"x", "y", "z", "ciao"};
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
    std::vector<std::string> colsnames = {"x", "y", "z", "ciao"};
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
    uint32_t max_items = 100;
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
    std::vector<std::string> colsnames = {"wordinfo"};
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
    std::vector<std::string> colsnames = {"ciao"};
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
    std::vector<std::string> read_colsnames = {"x", "y", "z", "ciao"};
    std::vector<std::string> write_colsnames = {"x", "y", "z"};
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
    uint32_t max_items = 100;
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
    std::vector<std::string> read_colsnames = {"x", "ciao"};
    std::vector<std::string> write_colsnames = {"x", "ciao"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";

    int64_t bigi= 9223372036854775807;
    std::vector<std::pair<int64_t, int64_t> > tokens = {
            std::pair<int64_t, int64_t>(-bigi -1,  -bigi/2),
            std::pair<int64_t, int64_t>(-bigi/2,0),
            std::pair<int64_t, int64_t>(0,bigi/2),
            std::pair<int64_t, int64_t>(bigi/2, bigi)
    };



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


/*

TEST(TestingPrefetch,Worker) {
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;
    uint32_t max_items = 100;
    const char * keyspace = "case18";
    const char * particles_table = "particle";
    const char * contact_p = "minerva-5";
    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, 9042);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

    const char* query = "SELECT * FROM particle WHERE partid=? AND time=?;";


    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(new CacheTable(max_items,particles_table,keyspace, query, test_session));


    const char* query_prefetch = "SELECT * FROM case18.particle WHERE token(partid)>? AND token(partid)<? ;";
    uint32_t num_ranges = 1;
    std::pair<uint64_t,uint64_t> *token_ranges = new std::pair<uint64_t ,uint64_t >[num_ranges* sizeof(std::pair<uint64_t,uint64_t>)];
    token_ranges[0].first=8070430489100699999;
    token_ranges[0].second=8070450532247928832;
    Prefetch* P = new Prefetch(token_ranges,max_items,T.get(),test_session,query_prefetch,1);
    uint16_t num_elem = 0;
    TupleRow *r_pref;
    while ((r_pref= P->get_next_tuple()) != NULL) {
        ++num_elem;
    };
    EXPECT_EQ(num_elem,84);
    delete(P);
}
*/



/*** CACHE API TESTS ***/
