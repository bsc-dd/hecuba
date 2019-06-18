#include <iostream>
#include <cassandra.h>
#include "gtest/gtest.h"
#include "../src/StorageDict.h"


using namespace std;

const char *keyspace = "test";
const char *particles_table = "particle";
const char *particles_wr_table = "particle_write";
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

    fireandforget("CREATE TABLE test.iface( key int, value double, PRIMARY KEY(key));", test_session);



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


/*** Bucket local tests ***/

TEST(TestingCPPIface, Bucket_compare_1) {
    Bucket<std::string,int> B1 = Bucket<std::string,int>(nullptr,"Hello");
    B1 = 123;

    Bucket<std::string,int> B2 = Bucket<std::string,int>(nullptr,"Hello");
    B2 = 124;

    EXPECT_GT(B2,B1);
    EXPECT_LT(B1,B2);
    EXPECT_NE(B1,B2);
    EXPECT_NE(B2,B1);


    B1 = 124;
    EXPECT_GE(B2,B1);
    EXPECT_LE(B1,B2);
    EXPECT_EQ(B1,B2);
    EXPECT_EQ(B2,B1);
}

/*** StorageDict local tests ***/

TEST(TestingCPPIface, StorageDict_local_construct_1) {
    StorageDict<int, std::string> SD = StorageDict<int, std::string>();
}


TEST(TestingCPPIface, StorageDict_local_construct_2) {
    StorageDict<double, std::vector<int> > SD = {};
}


TEST(TestingCPPIface, StorageDict_local_construct_3) {
    StorageDict<int, std::string> SD = {{123,"Firs text"},{456,"Second text"},{789, "Third test"}};
}



TEST(TestingCPPIface, StorageDict_local_insert_1) {
    StorageDict<int, std::string> SD = StorageDict<int, std::string>();
    SD.insert(std::pair<int,std::string>(1234567,"Some other text"));
}


TEST(TestingCPPIface, StorageDict_local_insert_2) {
    StorageDict<int, std::string> SD = StorageDict<int, std::string>();
    SD.insert(std::pair<int,std::string>(1234567,"Some other text"));
    SD.insert(std::pair<int,std::string>(1234567,"Totally different text"));
}


TEST(TestingCPPIface, StorageDict_local_access_1) {
    StorageDict<int, std::string> SD = {{123,"Firs text"},{456,"Second text"},{789, "Third test"}};
    std::string expected_text = "Second text";
    EXPECT_EQ(SD[456],expected_text);
    EXPECT_EQ(SD[789],"Third test");
    EXPECT_EQ(SD[123],"Firs text");
    int key = 852;
    std::string new_text = "A new inserted text";
    SD[key] = new_text;
    EXPECT_EQ(SD[key],new_text);
}


TEST(TestingCPPIface, StorageDict_local_access_2) {
    StorageDict<double, std::vector<int32_t > > SD;
    std::vector<int32_t > values;
    int key;
    for (key = 0; key<10; ++key) {
        values = {key*10,key*20,key*30};
        SD[key/10] = values;
    }
    std::vector<int32_t > results = SD[--key/10];
    EXPECT_EQ(results,values);
}


TEST(TestingCPPIface, StorageDict_local_access_3) {
    StorageDict<double, std::vector<int32_t > > SD;
    double key = 12.3;
    std::vector<int32_t > results = SD[key];
    std::vector<int32_t > values  = {};
    EXPECT_EQ(results,values);
}





/*** Tests with cassandra ***/


TEST(TestingCPPIface, Connection) {
    std::string nodeX = "127.0.0.1";
    //Configure Cassandra
    ClusterConfig *config = new ClusterConfig(9042, nodeX);
    EXPECT_EQ(config->get_contact_names(),nodeX);
    EXPECT_EQ(config->get_contact_port(),9042);
    delete(config);
}

TEST(TestingCPPIface, ConnectionClose) {
    std::string nodeX = "127.0.0.1";

    //Configure Cassandra
    ClusterConfig *config = new ClusterConfig(9042, nodeX);
    EXPECT_TRUE(config->get_session()!= nullptr);
    delete(config);
}





TEST(TestingCPPIface, StorageDict_use_case_1) {
    std::string nodeX = "127.0.0.1";

    //Configure Cassandra
    ClusterConfig *config = new ClusterConfig(9042, nodeX);

    //Use a non persistent dict
    int a = 123;
    std::string sentence = "Hello, World!";

    StorageDict<int, std::string> SD = StorageDict<int, std::string>(config);

    SD[a] = sentence; //Insert data
    EXPECT_EQ(SD[a], sentence);

    std::map<int, std::string> initialized_map = {{a,    "first sentence"},
                                                  {4546, "last_sentence"}};
    SD = initialized_map; //Assign a standard map from STL to our StorageDict

    EXPECT_EQ(SD[a], "first sentence");
    EXPECT_EQ(SD[4546], "last_sentence");

    std::string name_to_persist = "iface";
    SD.make_persistent(name_to_persist);

    SD[a] = sentence;
    EXPECT_EQ(SD[a], sentence);
    delete(config);
}



TEST(TestingCPPIface, StorageDict_use_case_2) {
    std::string nodeX = "127.0.0.1";

    //Configure Cassandra
    ClusterConfig *config = new ClusterConfig(9042, nodeX);

    //Use a non persistent dict
    int nelem = 1000;
    std::string sentence = "Hello, World!";

    StorageDict<int, double> *SD = new StorageDict<int, double >(config);

    std::string name_to_persist = "iface";
    SD->make_persistent(name_to_persist);


    for (int i = 0; i<nelem; ++i) {
        (*SD)[i] = i;
    }


    delete(SD);
    delete(config);
}

