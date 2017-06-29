#include <iostream>

#include <cassandra.h>
#include "gtest/gtest.h"
#include "../HCache.cpp"


using namespace std;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}




TEST(TestPyParse,SizeOfTypes) {
    Py_Initialize();
    PyEval_InitThreads();
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *key = PyLong_FromDouble(0.12);
    int ok = 0;

    size_t data_size=sizeof(cass_double_t);
    void *data_a = malloc(data_size);
    void *data_b = malloc(data_size);

    cass_double_t t;
    ok = PyArg_Parse(key, Py_DOUBLE, &t);
    EXPECT_EQ(ok,1);
    memcpy(data_a,&t,sizeof(t));
    ok = PyArg_Parse(key, Py_DOUBLE, data_b);
    EXPECT_EQ(ok,1);
    PyGILState_Release(gstate);

    EXPECT_EQ(memcmp(data_a,data_b,data_size),0);
    free(data_a);
    free(data_b);

}





const uint16_t i = 123;
const uint16_t j = 456;
/***
 * Testing comparators
 */

TEST(TupleTest, TupleOps) {
    size_t size=sizeof(uint16_t);
    auto sizes= vector<uint16_t>(2,size);
    char * buffer= (char *) malloc(size * 2);
    char * buffer2= (char *) malloc(size * 2);
    memcpy(buffer,&i,size);
    memcpy(buffer+size,&j,size);
    memcpy(buffer2,&i,size);
    memcpy(buffer2+size,&j,size);
    TupleRow t1 = TupleRow(&sizes,sizeof(uint16_t)*2,buffer);
    TupleRow t2 = TupleRow(&sizes,sizeof(uint16_t)*2,buffer2);

    //Equality
    EXPECT_TRUE(!(t1<t2)&&!(t2<t1));
    EXPECT_TRUE(!(t1>t2)&&!(t2>t1));


}



TEST(TestingCacheTable, SingleQ) {
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;
    uint32_t max_items = 100;
    const char * keyspace = "case18";
    const char * table = "particle";
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

    /** SETUP PY **/
    Py_Initialize();
    PyEval_InitThreads();
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    /** RANDOM  GENERATOR **/
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng{seed};

    // a distribution that takes randomness and produces values in specified range
    std::uniform_int_distribution<> dist(1,8000);
    /** KEYS **/
    uint32_t key1 = (uint32_t) dist(eng);
    float key2 = 0.003;

    PyObject* list = PyList_New(2);
    PyList_SetItem(list,0,Py_BuildValue("i",key1));
    PyList_SetItem(list,1,Py_BuildValue("f",key2));
    PyGILState_Release(gstate);
    std::vector<std::string> keysnames = {"partid","time"};
    std::vector<std::string> colsnames = {"x","xa","family"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t,int64_t > > tokens = {std::pair<int64_t,int64_t >(-10000,10000)};
   std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(new CacheTable(max_items,table,keyspace, keysnames,colsnames,token_pred,tokens, test_session));


    PyObject* result = T.get()->get_row(list);

    EXPECT_FALSE(result==0);
    gstate = PyGILState_Ensure();

    EXPECT_EQ(PyList_Size(result),colsnames.size());
    for ( int i = 0; i<PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None==PyList_GetItem(result,i));
    }
    PyGILState_Release(gstate);

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}



TEST(TestingCacheTable, PutRow) {
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;
    uint32_t max_items = 100;
    const char * keyspace = "case18";
    const char * table = "particle";
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

    /** SETUP PY **/
    Py_Initialize();
    PyEval_InitThreads();
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    /** RANDOM NUMBERS **/
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng{seed};

    // a distribution that takes randomness and produces values in specified range
    std::uniform_int_distribution<> dist(1,8000);
    /** KEYS **/
    uint32_t key1 = (uint32_t) dist(eng);
    float key2 = 0.003;

    PyObject* list = PyList_New(2);
    PyList_SetItem(list,0,Py_BuildValue("i",key1));
    PyList_SetItem(list,1,Py_BuildValue("f",key2));
    PyGILState_Release(gstate);
    std::vector<std::string> keysnames = {"partid","time"};
    std::vector<std::string> colsnames = {"x","xa","family"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t,int64_t > > tokens = {std::pair<int64_t,int64_t >(-10000,10000)};
    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(new CacheTable(max_items,table,keyspace, keysnames,colsnames,token_pred,tokens, test_session));

    EXPECT_FALSE(T==0);


    PyObject* result = T.get()->get_row(list);

    EXPECT_FALSE(result==0);
    gstate = PyGILState_Ensure();

    EXPECT_EQ(PyList_Size(result),colsnames.size());
    for ( int i = 0; i<PyList_Size(result); ++i) {
        EXPECT_FALSE(Py_None==PyList_GetItem(result,i));
    }
    PyGILState_Release(gstate);

    T.get()->put_row(list,result);

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}



TEST(TestingCacheTable, DeleteElem) {

    size_t ss=sizeof(uint16_t)*2;
    Poco::LRUCache<TupleRow, TupleRow> myCache(2);
    auto size = vector<uint16_t>(2, sizeof(uint16_t));
    char * b2= (char *) malloc(ss);
    memcpy(b2, &i, sizeof(uint16_t));
    memcpy(b2+sizeof(uint16_t ), &j, sizeof(uint16_t));
    TupleRow *t1 = new TupleRow(&size, sizeof(uint16_t) * 2,b2);

    uint16_t ka = 64;
    uint16_t kb = 128;
    b2= (char *) malloc(ss);
    memcpy(b2, &ka, sizeof(uint16_t));
    memcpy(b2 +sizeof(uint16_t), &kb, sizeof(uint16_t));

    TupleRow *key1 = new TupleRow(&size, sizeof(uint16_t) * 2,b2);
    myCache.add(*key1, t1);

    EXPECT_EQ(myCache.getAllKeys().size(), 1);
    TupleRow t = *(myCache.getAllKeys().begin());
    EXPECT_TRUE(t == *key1);
    EXPECT_FALSE(&t == key1);
    /**
     * Reason: myCache builds its own copy of key1 through the copy constructor. They are equal but not the same object
     **/
    myCache.clear();
    //Removes all references, and deletes all objects. Key1 is still active thanks to our ref
    delete(key1);
}



TEST(TestingCacheTable, MultiQ) {
    /** CONNECT **/
    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;
    uint32_t max_items = 100;
    const char * keyspace = "case18";
    const char * table = "particle";
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

/** INITIALIZE PYTHON **/
    Py_Initialize();
    PyEval_InitThreads();
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* list = PyList_New(2);
    PyList_SetItem(list,0,Py_BuildValue("i",123));
    PyList_SetItem(list,1,Py_BuildValue("f",0.003));
    PyObject* list2 = PyList_New(2);
    PyList_SetItem(list2,0,Py_BuildValue("i",543));
    PyList_SetItem(list2,1,Py_BuildValue("f",0.003));

    PyObject* list3 = PyList_New(2);
    PyList_SetItem(list3,0,Py_BuildValue("i",323));
    PyList_SetItem(list3,1,Py_BuildValue("f",0.003));
    PyGILState_Release(gstate);


    std::vector<std::string> keysnames = {"partid","time"};
    std::vector<std::string> colsnames = {"x","xa","family"};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t,int64_t > > tokens = {std::pair<int64_t,int64_t >(-10000,10000)};
    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(new CacheTable(max_items,table,keyspace, keysnames,colsnames,token_pred,tokens, test_session));

    PyObject* result = T.get()->get_row(list);

    result = T.get()->get_row(list2);
    result = T.get()->get_row(list3);
    result = T.get()->get_row(list);
    result = T.get()->get_row(list2);
    result = T.get()->get_row(list3);
    result = T.get()->get_row(list);

    gstate = PyGILState_Ensure();

    for ( int i = 0; i<PyList_Size(result); ++i) {
        PyList_GetItem(result,i);
    }
    PyGILState_Release(gstate);


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
    const char * table = "particle";
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


    std::shared_ptr<CacheTable> T = std::shared_ptr<CacheTable>(new CacheTable(max_items,table,keyspace, query, test_session));


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