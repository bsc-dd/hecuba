#include <iostream>
#include <cassandra.h>
#include "gtest/gtest.h"
#include "../CacheTable.h"
#include "../StorageInterface.h"

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



    fireandforget("CREATE TABLE test.bytes(partid int PRIMARY KEY, data blob);", test_session);
    fireandforget("CREATE TABLE test.arrays(partid int PRIMARY KEY, image uuid);", test_session);
    fireandforget("CREATE TABLE test.arrays_aux(uuid uuid,  position int, data blob, PRIMARY KEY (uuid,position));", test_session);


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
    /*
    PyObject *key = PyInt_FromSize_t(123);
    PyObject *value = PyInt_FromSize_t(233);
    tbb::concurrent_bounded_queue<std::pair<PyObject *, PyObject *> > data;
    data.push(std::make_pair(key, value));

    std::pair<PyObject *, PyObject *> item;
    EXPECT_TRUE(data.try_pop(item));
    EXPECT_EQ(PyInt_AsLong(item.first), 123);
    EXPECT_EQ(PyInt_AsLong(item.second), 233);
*/
     }


/** Test to asses Poco Cache is performing as expected with pointer **/
TEST(TestingPocoCache, InsertGetDeleteOps) {
    const uint16_t i = 123;
    const uint16_t j = 456;
    size_t ss = sizeof(uint16_t) * 2;
    Poco::LRUCache<TupleRow, TupleRow> myCache(2);

    ColumnMeta cm1={{"ciao"},CASS_VALUE_TYPE_INT,0,sizeof(uint16_t)};
    ColumnMeta cm2 ={{"ciaociao"},CASS_VALUE_TYPE_INT,sizeof(uint16_t),sizeof(uint16_t)};
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


    ColumnMeta cm1={{"ciao"},CASS_VALUE_TYPE_INT,0,sizeof(uint16_t)};
    ColumnMeta cm2 ={{"ciaociao"},CASS_VALUE_TYPE_INT,sizeof(uint16_t),sizeof(uint16_t)};
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
    int32_t size = sizeof(uint16_t);
    auto sizes = vector<uint16_t>(2, size);
    char *buffer = (char *) malloc(size * 2);
    char *buffer2 = (char *) malloc(size * 2);
    memcpy(buffer, &i, size);
    memcpy(buffer + size, &j, size);
    memcpy(buffer2, &i, size);
    memcpy(buffer2 + size, &j, size);


    ColumnMeta cm1={{"ciao"},CASS_VALUE_TYPE_INT,0,sizeof(uint16_t)};
    ColumnMeta cm2 ={{"ciaociao"},CASS_VALUE_TYPE_INT,sizeof(uint16_t),sizeof(uint16_t)};
    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);



    TupleRow t1 = TupleRow(metas,sizeof(uint16_t) * 2, buffer);
    TupleRow t2 = TupleRow(metas,sizeof(uint16_t) * 2, buffer2);

    //Equality
    EXPECT_TRUE(!(t1 < t2) && !(t2 < t1));
    EXPECT_TRUE(!(t1 > t2) && !(t2 > t1));
    cm2 ={{"ciaocia"},CASS_VALUE_TYPE_INT,sizeof(uint16_t),sizeof(uint16_t)};



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
    cass_cluster_set_port(test_cluster, nodePort);

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
    std::vector<  std::vector<std::string> >colsnames = { std::vector<std::string>{"x"},  std::vector<std::string>{"y"},  std::vector<std::string>{"z"},  std::vector<std::string>{"ciao"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};
    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    TableMetadata* table_meta = new TableMetadata(particles_table,keyspace,keysnames,colsnames,test_session);

    CacheTable *CTable = new CacheTable(table_meta, test_session, config);

    TupleRow *t = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);


    const TupleRow *result = CTable->get_crow(t)[0];

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
    cass_cluster_set_port(test_cluster, nodePort);

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
    std::vector<  std::vector<std::string> >colsnames = { std::vector<std::string>{"ciao"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    TableMetadata* table_meta = new TableMetadata(particles_table,keyspace,keysnames,colsnames,test_session);

    CacheTable T = CacheTable(table_meta, test_session, config);

    TupleRow *t = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);

    const TupleRow *result = T.get_crow(t)[0];

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
//Replacement inside cache is broken, the payload is being freed twice (once on replace and another thereafter)
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

    char *buffer = (char *) malloc(sizeof(int) + sizeof(float));
    int val = 1234;
    memcpy(buffer, &val, sizeof(int));
    float f = 12340;
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    std::vector<uint16_t> offsets = {0, sizeof(int)};


    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<  std::vector<std::string> >colsnames = { std::vector<std::string>{"ciao"}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata* table_meta = new TableMetadata(particles_table,keyspace,keysnames,colsnames,test_session);

    CacheTable* T = new CacheTable(table_meta, test_session, config);


    //TupleRow *t = new TupleRow(T._test_get_keys_factory()->get_metadata(), sizeof(int) + sizeof(float), buffer);

    std::shared_ptr<void> result = T->get_crow(buffer);

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


    buffer = (char *) malloc(sizeof(int) + sizeof(float));
    memcpy(buffer, &val, sizeof(int));
    memcpy(buffer + sizeof(int), &f, sizeof(float));

     T->put_crow(buffer,result.get());

    delete(T);
    //With the aim to synchronize
    T = new CacheTable(table_meta, test_session, config);


    buffer = (char *) malloc(sizeof(int) + sizeof(float));
    memcpy(buffer, &val, sizeof(int));
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    result = T->get_crow(buffer);


    EXPECT_FALSE(result == NULL);

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


    buffer = (char *) malloc(sizeof(int) + sizeof(float));
    memcpy(buffer, &val, sizeof(int));
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    T->put_crow(buffer,result.get());

    delete(T);
    //With the aim to synchronize
    T = new CacheTable(table_meta, test_session, config);


    buffer = (char *) malloc(sizeof(int) + sizeof(float));
    memcpy(buffer, &val, sizeof(int));
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    result = T->get_crow(buffer);
    if (result != 0) {

        const void *v = result.get();
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "74040");
    }
    delete(T);

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
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);
    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

    char *buffer = (char *) malloc(sizeof(int) + sizeof(float));

    int val = 1234;
    memcpy(buffer, &val, sizeof(int));

    float f = 12340;
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    ColumnMeta cm1={{"partid"},CASS_VALUE_TYPE_INT,0,sizeof(int)};
    ColumnMeta cm2 ={{"time"},CASS_VALUE_TYPE_FLOAT,sizeof(int),sizeof(float)};

    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<  std::vector<std::string> >colsnames = { std::vector<std::string>{"ciao"}};
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
    config["prefetch_size"]="100";


    TableMetadata* table_meta = new TableMetadata(particles_table,keyspace,keysnames,colsnames,test_session);

    CacheTable T = CacheTable(table_meta, test_session, config);

    Prefetch *P = new Prefetch(tokens,table_meta,test_session,config);


    TupleRow *result = P->get_cnext();
    EXPECT_FALSE(result == NULL);
    uint16_t it = 1;


    while ((result = P->get_cnext())!=NULL) {
        const void *v = result->get_element(2);
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





/*** CACHE API TESTS ***/


TEST(TestingStorageInterfaceCpp,ConnectDisconnect){
    StorageInterface* StorageI= new StorageInterface(nodePort,contact_p);
    delete(StorageI);
}




TEST(TestingStorageInterfaceCpp,CreateAndDelCache){
    /** KEYS **/

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<  std::vector<std::string> >read_colsnames = {std::vector<std::string>{"x"}, std::vector<std::string>{"ciao"}};
    std::vector<  std::vector<std::string> >write_colsnames = {std::vector<std::string>{"y"}, std::vector<std::string>{"ciao"}};
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

    StorageInterface *StorageI = new StorageInterface(nodePort,contact_p);
    CacheTable* table= StorageI->make_cache(particles_table, keyspace, keysnames, read_colsnames, config);

    delete(table);
    delete(StorageI);
}



TEST(TestingStorageInterfaceCpp,CreateAndDelCacheWrong){
    /** This test demonstrates that deleting the Cache provider
     * before deleting cache instances doesnt raise exceptions
     * or enter in any kind of lock
     * **/

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<  std::vector<std::string> >read_colsnames = {std::vector<std::string>{"x"}, std::vector<std::string>{"ciao"}};

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

    StorageInterface *StorageI = new StorageInterface(nodePort,contact_p);
    CacheTable* table= StorageI->make_cache(particles_table, keyspace, keysnames, read_colsnames, config);

    delete(StorageI);
}


TEST(TestingStorageInterfaceCpp,IteratePrefetch){
    /** This test demonstrates that deleting the Cache provider
     * before deleting cache instances doesnt raise exceptions
     * or enter in any kind of lock
     * **/

    std::vector<std::string> keysnames = {"partid", "time"};
    std::vector<  std::vector<std::string> >read_colsnames = {std::vector<std::string>{"x"}, std::vector<std::string>{"ciao"}};
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
    config["prefetch_size"] = "100";

    StorageInterface *StorageI = new StorageInterface(nodePort,contact_p);
    CacheTable* table= StorageI->make_cache(particles_table, keyspace, keysnames, read_colsnames, config);

    Prefetch *P = StorageI->get_iterator(particles_table, keyspace, keysnames, read_colsnames, tokens,config);
    int it= 0;
    while (P->get_cnext()) {
        ++it;
    }
    EXPECT_EQ(it,10001);
    delete(P);
    delete(table);
    delete(StorageI);
}