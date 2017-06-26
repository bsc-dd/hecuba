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

    if (test_session != NULL) {
        CassFuture *close_future = cass_session_close(test_session);
        cass_future_free(close_future);
        cass_session_free(test_session);
        cass_cluster_free(test_cluster);
        test_session = NULL;
    }

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

    ColumnMeta cm1=ColumnMeta();
    cm1.info={{"name","ciao"}};
    cm1.type=CASS_VALUE_TYPE_INT;
    cm1.position=0;
    cm1.size=sizeof(uint16_t);

    ColumnMeta cm2=ColumnMeta();
    cm2.info={{"name","ciaociao"}};
    cm2.type=CASS_VALUE_TYPE_INT;
    cm2.position=sizeof(uint16_t);
    cm2.size=sizeof(uint16_t);

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

    ColumnMeta cm1=ColumnMeta();
    cm1.info={{"name","ciao"}};
    cm1.type=CASS_VALUE_TYPE_INT;
    cm1.position=0;
    cm1.size=sizeof(uint16_t);

    ColumnMeta cm2=ColumnMeta();
    cm2.info={{"name","ciaociao"}};
    cm2.type=CASS_VALUE_TYPE_INT;
    cm2.position=sizeof(uint16_t);
    cm2.size=sizeof(uint16_t);

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
TEST(TupleTest, TestNulls) {
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

    ColumnMeta cm1=ColumnMeta();
    cm1.info={{"name","ciao"}};
    cm1.type=CASS_VALUE_TYPE_INT;
    cm1.position=0;
    cm1.size=sizeof(uint16_t);

    ColumnMeta cm2=ColumnMeta();
    cm2.info={{"name","ciaociao"}};
    cm2.type=CASS_VALUE_TYPE_INT;
    cm2.position=sizeof(uint16_t);
    cm2.size=sizeof(uint16_t);

    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);



    TupleRow t1 = TupleRow(metas,sizeof(uint16_t) * 2, buffer);
    TupleRow t2 = TupleRow(metas,sizeof(uint16_t) * 2, buffer2);

    //Equality
    EXPECT_TRUE(!(t1 < t2) && !(t2 < t1));
    EXPECT_TRUE(!(t1 > t2) && !(t2 > t1));

    //same position null, they are equal
    t1.setNull(1);
    t2.setNull(1);
    EXPECT_TRUE(!(t1 < t2) && !(t2 < t1));
    EXPECT_TRUE(!(t1 > t2) && !(t2 > t1));
    //one position is null they differ
    t1.unsetNull(1);
    EXPECT_FALSE(!(t1 < t2) && !(t2 < t1));
    EXPECT_FALSE(!(t1 > t2) && !(t2 > t1));
    EXPECT_TRUE(t1<t2);// And the one with nulls is smaller
    //they both have all elements set to non null and they are equal again
    t2.unsetNull(1);
    EXPECT_TRUE(!(t1 < t2) && !(t2 < t1));
    EXPECT_TRUE(!(t1 > t2) && !(t2 > t1));
    //setting different positions to null make them differ
    t1.setNull(1);
    t2.setNull(0);
    EXPECT_FALSE(!(t1 < t2) && !(t2 < t1));
    EXPECT_FALSE(!(t1 > t2) && !(t2 > t1));
    //and t2 should be smaller than t1 since it has a smaller element null
    EXPECT_TRUE(t2<t1);
    //they have all elements to null but t2 has position 1 to valid
    t1.setNull(0);
    EXPECT_FALSE(!(t1 < t2) && !(t2 < t1));
    EXPECT_FALSE(!(t1 > t2) && !(t2 > t1));
    EXPECT_TRUE(t2<t1);
    //All elements are null, they must be equal
    t2.setNull(1);
    EXPECT_TRUE(!(t1 < t2) && !(t2 < t1));
    EXPECT_TRUE(!(t1 > t2) && !(t2 > t1));
    //No nulls in any tuple, they mjust be equal
    t1.unsetNull(0);
    t1.unsetNull(1);
    t2.unsetNull(0);
    t2.unsetNull(1);
    EXPECT_TRUE(!(t1 < t2) && !(t2 < t1));
    EXPECT_TRUE(!(t1 > t2) && !(t2 > t1));
    //getting an element no null returns a valid ptr
    EXPECT_FALSE(t1.get_element(1)==nullptr);
    //getting a null element returns a null ptr
    t1.setNull(1);
    EXPECT_TRUE(t1.get_element(1)==nullptr);
    //however, the other tuple still returns a valid ptr for the same position
    EXPECT_FALSE(t2.get_element(1)==nullptr);
    t1.unsetNull(1);
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

    ColumnMeta cm1=ColumnMeta();
    cm1.info={{"name","ciao"}};
    cm1.type=CASS_VALUE_TYPE_INT;
    cm1.position=0;
    cm1.size=sizeof(uint16_t);

    ColumnMeta cm2=ColumnMeta();
    cm2.info={{"name","ciaociao"}};
    cm2.type=CASS_VALUE_TYPE_INT;
    cm2.position=sizeof(uint16_t);
    cm2.size=sizeof(uint16_t);

    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);



    TupleRow t1 = TupleRow(metas,sizeof(uint16_t) * 2, buffer);
    TupleRow t2 = TupleRow(metas,sizeof(uint16_t) * 2, buffer2);

    //Equality
    EXPECT_TRUE(!(t1 < t2) && !(t2 < t1));
    EXPECT_TRUE(!(t1 > t2) && !(t2 > t1));


    cm2=ColumnMeta();
    cm2.info={{"name","ciaociao"}};
    cm2.type=CASS_VALUE_TYPE_INT;
    cm2.position=sizeof(uint16_t);
    cm2.size=sizeof(uint16_t);


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


    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> >colsnames = { {{"name", "x"}},  {{"name", "y"}},  {{"name", "z"}},
                                                                  {{"name", "ciao"}}};
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





TEST(TestingCacheTable, StoreNull) {

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


//partid int,time float,x float,y float,z float, PRIMARY KEY(partid,time)

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> > colsnames = { {{"name", "x"}},{{"name", "y"}},{{"name", "z"}}};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata* table_meta = new TableMetadata(particles_wr_table,keyspace,keysnames,colsnames,test_session);

    CacheTable* cache = new CacheTable(table_meta, test_session, config);


    char *buffer = (char *) malloc(sizeof(int)+sizeof(float)); //keys

    int32_t k1 = 3682;
    float k2 = 143.2;
    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    float v[] = {0.01, 1.25, 0.98};

    char *buffer2 = (char *) malloc(sizeof(float)*3); //values
    memcpy(buffer2, &v, sizeof(float)*3);


    TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(int)+sizeof(float), buffer);
    TupleRow *values = new TupleRow(table_meta->get_values(), sizeof(float)*3, buffer2);


    values->setNull(1);

    cache->put_crow(keys, values);

    delete(keys);
    delete(values);
    delete(cache);

    /*** now we write into another table with text ***/

    keysnames = {{{"name", "partid"}},{{"name","time"}}};
    colsnames = { {{"name", "x"}},{{"name", "ciao"}}};

    tokens = {};



    table_meta = new TableMetadata(words_wr_table,keyspace,keysnames,colsnames,test_session);

    cache = new CacheTable(table_meta, test_session, config);


    buffer = (char *) malloc(sizeof(int)+sizeof(float)); //keys

    k1 = 3682;
    k2 = 143.2;
    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    float v1 = 23.12;
    std::string *v2 = new std::string("someBeautiful, and randomText");

    buffer2 = (char *) malloc(sizeof(float)+sizeof(char*)); //values
    memcpy(buffer2, &v1, sizeof(float));
    memcpy(buffer2+sizeof(float), &v2, sizeof(char *));


    keys = new TupleRow(table_meta->get_keys(), sizeof(int)+sizeof(float), buffer);
    values = new TupleRow(table_meta->get_values(), sizeof(float)*3, buffer2);


    values->setNull(1);

    cache->put_crow(keys, values);

    delete(keys);
    delete(values);
    delete(cache);

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}



TEST(TestingCacheTable, StoreNullBulk) {

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


//partid int,time float,x float,y float,z float, PRIMARY KEY(partid,time)

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> > colsnames = { {{"name", "x"}},{{"name", "y"}},{{"name", "z"}}};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata* table_meta = new TableMetadata(particles_wr_table,keyspace,keysnames,colsnames,test_session);

    CacheTable* cache = new CacheTable(table_meta, test_session, config);

    for (uint32_t id = 0; id<8000; ++id ) {
        char *buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

        int32_t k1 = 3682;
        float k2 = 143.2;
        memcpy(buffer, &id, sizeof(int));
        memcpy(buffer + sizeof(int), &k2, sizeof(float));

        float v[] = {0.01, 1.25, 0.98};

        char *buffer2 = (char *) malloc(sizeof(float) * 3); //values
        memcpy(buffer2, &v, sizeof(float) * 3);


        TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);
        TupleRow *values = new TupleRow(table_meta->get_values(), sizeof(float) * 3, buffer2);


        values->setNull(1);

        cache->put_crow(keys, values);
        delete(keys);
        delete(values);
    }

    delete(cache);


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}



TEST(TestingCacheTable, StoreNullBulkText) {

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


//partid int,time float,x float,y float,z float, PRIMARY KEY(partid,time)

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> > colsnames = {{{"name", "x"}},{{"name", "ciao"}}};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata* table_meta = new TableMetadata(words_wr_table,keyspace,keysnames,colsnames,test_session);

    CacheTable* cache = new CacheTable(table_meta, test_session, config);

    for (uint32_t id = 0; id<8000; ++id ) {
        char *buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

        float k2 = 143.2;
        memcpy(buffer, &id, sizeof(int));
        memcpy(buffer + sizeof(int), &k2, sizeof(float));

        float v[] = {0.01, 1.25, 0.98};

        char *buffer2 = (char *) malloc(sizeof(float)+sizeof(char*)); //values
        memcpy(buffer2, &v, sizeof(float));


        TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);
        TupleRow *values = new TupleRow(table_meta->get_values(), sizeof(float) * 3, buffer2);


        values->setNull(1);

        cache->put_crow(keys, values);
        delete(keys);
        delete(values);
    }

    delete(cache);


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}





TEST(TestingCacheTable, ReadNull) {

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


//partid int,time float,x float,y float,z float, PRIMARY KEY(partid,time)

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> > colsnames = { {{"name", "x"}},{{"name", "y"}},{{"name", "z"}}};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata* table_meta = new TableMetadata(particles_wr_table,keyspace,keysnames,colsnames,test_session);

    CacheTable* cache = new CacheTable(table_meta, test_session, config);


    char *buffer = (char *) malloc(sizeof(int)+sizeof(float)); //keys

    int32_t k1 = 4682;
    float k2 = 93.2;
    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    float v[] = {4.43, 9.99, 1.238};

    char *buffer2 = (char *) malloc(sizeof(float)*3); //values
    memcpy(buffer2, &v, sizeof(float)*3);


    TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(int)+sizeof(float), buffer);
    TupleRow *values = new TupleRow(table_meta->get_values(), sizeof(float)*3, buffer2);


    values->setNull(1);

    cache->put_crow(keys, values);

    delete(keys);
    delete(values);
    delete(cache);


    //now we read the data

    keysnames = {{{"name", "partid"}},{{"name","time"}}};
    colsnames = { {{"name", "x"}},{{"name", "y"}},{{"name", "z"}}};

    tokens = {};

    table_meta = new TableMetadata(particles_wr_table,keyspace,keysnames,colsnames,test_session);

    cache = new CacheTable(table_meta, test_session, config);


    buffer = (char *) malloc(sizeof(int)+sizeof(float)); //keys

    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    keys = new TupleRow(table_meta->get_keys(), sizeof(int)+sizeof(float), buffer);


    std::vector <const TupleRow*> results = cache->get_crow(keys);
    ASSERT_TRUE(results.size()==1);
    const TupleRow* read_values = results[0];
    const float *v0 = (float*) read_values->get_element(0);
    const float *v1 = (float*) read_values->get_element(1);
    const float *v2 = (float*) read_values->get_element(2);
    EXPECT_DOUBLE_EQ(*v0,v[0]);
    EXPECT_TRUE(v1== nullptr);
    EXPECT_DOUBLE_EQ(*v2,v[2]);

    delete(keys);
    delete(read_values);
    delete(cache);

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


    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> >colsnames = {{{"name", "ciao"}}};

    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    TableMetadata* table_meta = new TableMetadata(particles_table,keyspace,keysnames,colsnames,test_session);

    CacheTable* cache = new CacheTable(table_meta, test_session, config);


    TupleRow *t = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);

    std::vector <const TupleRow*> all_rows = cache->get_crow(t);
    delete(t);
    delete(cache);
    EXPECT_EQ(all_rows.size(),1);
    const TupleRow *result = all_rows.at(0);

    EXPECT_FALSE(result == NULL);

    if (result != 0) {

        const void *v = result->get_element(0);
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "74040");

    }

    for (const TupleRow* tuple:all_rows) {
        delete(tuple);
    }

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}




/** Test there are no problems when requesting twice the same key **/
TEST(TestingCacheTable, GetRowStringSameKey) {

    uint32_t  n_queries = 100;


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

    /** setup cache **/

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> >colsnames = {{{"name", "ciao"}}};


    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    TableMetadata* table_meta = new TableMetadata(particles_table,keyspace,keysnames,colsnames,test_session);
    CacheTable* cache = new CacheTable(table_meta, test_session, config);


    /** build key **/

    char *buffer = (char *) malloc(sizeof(int) + sizeof(float));

    int val = 1234;
    memcpy(buffer, &val, sizeof(int));

    float f = 12340;
    memcpy(buffer + sizeof(int), &f, sizeof(float));
    TupleRow *key = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);


    /** get queries **/

    for (uint32_t query_i = 0; query_i<n_queries; query_i++) {

        std::vector <const TupleRow*> all_rows = cache->get_crow(key);
        EXPECT_EQ(all_rows.size(),1);

        const TupleRow *result = all_rows.at(0);
        EXPECT_FALSE(result == NULL);

        if (result != 0) {
            const void *v = result->get_element(0);
            int64_t addr;
            memcpy(&addr, v, sizeof(char *));
            char *d = reinterpret_cast<char *>(addr);

            EXPECT_STREQ(d, "74040");
        }

        for (const TupleRow* tuple:all_rows) {
            delete(tuple);
        }
    }


    delete(key);
    delete(cache);
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

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> >colsnames = { {{"name", "ciao"}}};

    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata* table_meta = new TableMetadata(particles_table,keyspace,keysnames,colsnames,test_session);

    CacheTable* T = new CacheTable(table_meta, test_session, config);



    //TupleRow *t = new TupleRow(T._test_get_keys_factory()->get_metadata(), sizeof(int) + sizeof(float), buffer);

    std::vector<const TupleRow*> results = T->get_crow(buffer);


    EXPECT_FALSE(results.empty());

    if (!results.empty()) {

        const void *v = results[0]->get_payload();
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "74040");
    }

    buffer = (char *) malloc(sizeof(int) + sizeof(float));
    memcpy(buffer, &val, sizeof(int));
    memcpy(buffer + sizeof(int), &f, sizeof(float));


    char *substitue = (char*) malloc(sizeof("71919"));
    memcpy(substitue,"71919",sizeof("71919"));
    char** payload2 = (char**)malloc(sizeof(char*));
    *payload2=substitue;

    T->put_crow(buffer,payload2);


    delete(T);
    //With the aim to synchronize
    table_meta = new TableMetadata(particles_table,keyspace,keysnames,colsnames,test_session);
    T = new CacheTable(table_meta, test_session, config);


    buffer = (char *) malloc(sizeof(int) + sizeof(float));
    memcpy(buffer, &val, sizeof(int));
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    results = T->get_crow(buffer);


    EXPECT_FALSE(results.empty());

    if (!results.empty()) {
        const void *v = results[0]->get_payload();
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "71919");
    }

    substitue = (char*) malloc(sizeof("74040"));
    memcpy(substitue,"74040",sizeof("74040"));
    payload2 = (char**)malloc(sizeof(char*));
    *payload2=substitue;

    buffer = (char *) malloc(sizeof(int) + sizeof(float));
    memcpy(buffer, &val, sizeof(int));
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    T->put_crow(buffer,payload2);

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

    ColumnMeta cm1=ColumnMeta();
    cm1.info={{"name","partid"}};
    cm1.type=CASS_VALUE_TYPE_INT;
    cm1.position=0;
    cm1.size=sizeof(int);

    ColumnMeta cm2=ColumnMeta();
    cm2.info={{"name","time"}};
    cm2.type=CASS_VALUE_TYPE_FLOAT;
    cm2.position=sizeof(int);
    cm2.size=sizeof(float);

    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> >colsnames = { {{"name", "ciao"}}};

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



TEST(TestingPrefetch, GetNextAndUpdateCache) {
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

    ColumnMeta cm1=ColumnMeta();
    cm1.info={{"name","partid"}};
    cm1.type=CASS_VALUE_TYPE_INT;
    cm1.position=0;
    cm1.size=sizeof(int);

    ColumnMeta cm2=ColumnMeta();
    cm2.info={{"name","time"}};
    cm2.type=CASS_VALUE_TYPE_FLOAT;
    cm2.position=sizeof(int);
    cm2.size=sizeof(float);

    std::vector<ColumnMeta> v = {cm1,cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas=std::make_shared<std::vector<ColumnMeta>>(v);

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> >colsnames = { {{"name", "ciao"}}};

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
    config["cache_size"] = "100";
    config["prefetch_size"]="30";
    config["update_cache"]="true";

    TableMetadata* table_meta = new TableMetadata(particles_table,keyspace,keysnames,colsnames,test_session);

    CacheTable T = CacheTable(table_meta, test_session, config);

    //By default iterates items
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



TEST(TestingStorageInterfaceCpp,CreateAndDelCache){
    /** KEYS **/

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> >read_colsnames = {{{"name","x"}}, {{"name", "ciao"}}};
    std::vector< std::map<std::string,std::string> >write_colsnames = {{{"name","y"}}, {{"name", "ciao"}}};

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

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> >read_colsnames = {{{"name","x"}}, {{"name", "ciao"}}};


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
    delete(table);
}


TEST(TestingStorageInterfaceCpp,IteratePrefetch){
    /** This test demonstrates that deleting the Cache provider
     * before deleting cache instances doesnt raise exceptions
     * or enter in any kind of lock
     * **/

    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> >read_colsnames = {{{"name","x"}},{{"name", "ciao"}}};

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
