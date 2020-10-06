#include <iostream>
#include <cassandra.h>
#include "gtest/gtest.h"
#include "../src/CacheTable.h"
#include "../src/StorageInterface.h"

using namespace std;

#define PY_ERR_CHECK if (PyErr_Occurred()){PyErr_Print(); PyErr_Clear();}

const char *keyspace = "test";
const char *particles_table = "particle";
const char *particles_wr_table = "particle_write";
const char *words_wr_table = "words_write";
const char *only_keys_table = "only_keys";
const char *contact_p = "127.0.0.1";

uint32_t nodePort = 9042;

const CassResult* fireandgetresult(const char *query, CassSession *session) {
    std::cout << "EXECUTING " << query << std::endl;
    CassStatement *statement = cass_statement_new(query, 0);
    CassFuture *query_future = cass_session_execute(session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    CassError rc = cass_future_error_code(query_future);
    if (result == NULL) {
        std::string error(cass_error_desc(rc));
        cass_future_free(query_future);
        cass_statement_free(statement);
        throw ModuleException("CacheTable: Get row error on result" + error);
    }
    return cass_future_get_result(query_future);
}

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

void fireandforgetstatement(CassStatement *statement, CassSession *session) {
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

    for (int i = 0; i <= 100; i++) {
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

    fireandforget(
            "CREATE TABLE test.only_keys(first int, second int, third text, PRIMARY KEY((first, second), third));",
            test_session);

    fireandforget(
            "CREATE TABLE test.particle_write( partid int,time float,x float,y float,z float, PRIMARY KEY(partid,time));",
            test_session);


    fireandforget(
            "CREATE TABLE test.words_write( partid int,time float, x float, ciao text, PRIMARY KEY(partid,time));",
            test_session);


    fireandforget("CREATE TABLE test.bytes(partid int PRIMARY KEY, data blob);", test_session);
    fireandforget("CREATE TABLE test.arrays(partid int PRIMARY KEY, image uuid);", test_session);
    fireandforget("CREATE TABLE test.arrays_aux(uuid uuid,  position int, data blob, PRIMARY KEY (uuid,position));",
                  test_session);

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

TEST(StdCassandra, StdCassandra_test1) {

    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);

    /*
    std::vector<std::map<std::string, std::string> > keys_arrow_names = {{{"name", "storage_id"}},
                                                                         {{"name", "col_id"}}};
    std::vector<std::map<std::string, std::string> > columns_buffer_names = {{{"name", "row_id"}},
                                                                             {{"name", "size_elem"}},
                                                                             {{"name", "payload"}}};
    std::vector<std::map<std::string, std::string> > columns_arrow_names = {{{"name", "arrow_addr"}},
                                                                            {{"name", "arrow_size"}}};
    TableMetadata *table_meta_buffer = new TableMetadata("people", "test", keys_arrow_names, columns_buffer_names, test_session);
    TableMetadata *table_meta_arrow = new TableMetadata("people", "test", keys_arrow_names, columns_arrow_names, test_session);
     */


    CassError rc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rc == CASS_OK);

    cass_future_free(connect_future);

    fireandforget("DROP KEYSPACE IF EXISTS test;", test_session);
    fireandforget("DROP KEYSPACE IF EXISTS test_arrow;", test_session);

    fireandforget("CREATE KEYSPACE IF NOT EXISTS test_arrow WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};",
    test_session);

    fireandforget(
    "CREATE TABLE test_arrow.integers_buffer (storage_id uuid, col_id bigint, row_id bigint, size_elem int, payload blob, PRIMARY KEY (storage_id, col_id)               );",
    test_session);

    fireandforget(
    "CREATE TABLE test_arrow.integers_arrow (storage_id uuid, col_id bigint, arrow_addr bigint, arrow_size int, PRIMARY KEY (storage_id, col_id)               );",
    test_session);

    CassStatement* statement = NULL;
    const char* query = "INSERT INTO test_arrow.integers_buffer (storage_id, col_id, row_id, size_elem, payload) VALUES (?, ?, ?, ?, ?);";
    for (int i = 0; i < 6; ++i) {
        statement = cass_statement_new(query, 5);
        CassUuid uuid = {};
        cass_uuid_from_string("c37d661d-7e61-49ea-96a5-68c34e83db3a", &uuid);
        cass_statement_bind_uuid(statement, 0, uuid);
        cass_statement_bind_int64(statement, 1, (cass_int64_t)i);
        cass_statement_bind_int64(statement, 2, (cass_int64_t)1);
        cass_statement_bind_int32(statement, 3, (cass_int32_t)8);

        uint64_t payload = i;
        std::unique_ptr<char[]> buffer(new char[sizeof(uint64_t)]);
        memcpy(buffer.get(), &payload, sizeof(uint64_t));
        cass_statement_bind_bytes(statement, 4, (cass_byte_t*)buffer.get(), sizeof(uint64_t));

        fireandforgetstatement(statement, test_session);
    }

    //NOW LET'S TRY TO RETRIEVE THE DATA INSERTED
    //No metada since there isn't hecuba right now
    const CassResult *result = fireandgetresult("SELECT * FROM test_arrow.integers_buffer;", test_session);
    /*
    std::vector<const TupleRow *> gvalues(cass_result_row_count(result));
    const CassRow *row;
    CassIterator *it = cass_iterator_from_result(result);
    int64_t counter = 0;
    std::shared_ptr<const std::vector<ColumnMeta> > cols = table_meta_buffer->get_values();
    TupleRowFactory trf = TupleRowFactory(cols);
    while (cass_iterator_next(it)) {
        row = cass_iterator_get_row(it);
        gvalues[counter] = trf.make_tuple(row);
        ++counter;
    }
    EXPECT_EQ(counter, 6); */


    CassIterator* rows = cass_iterator_from_result(result);
    int i = 0;
    while (cass_iterator_next(rows)) {
        const CassRow* row = cass_iterator_get_row(rows);
        const CassValue* cassValueColdId = cass_row_get_column_by_name(row, "col_id");
        cass_int64_t colId;
        cass_value_get_int64(cassValueColdId, &colId);
        EXPECT_EQ(colId, i);
        ++i;
    }
    EXPECT_EQ(i, 6);


    result = fireandgetresult("SELECT * FROM test_arrow.integers_arrow;", test_session);
    rows = cass_iterator_from_result(result);
    i = 0;
    while (cass_iterator_next(rows)) {
        const CassRow* row = cass_iterator_get_row(rows);
        const CassValue* cassValueAddr = cass_row_get_column_by_name(row, "arrow_addr");
        cass_int64_t addr;
        cass_value_get_int64(cassValueAddr, &addr);
        EXPECT_EQ(addr, i*8);
        ++i;
    }
    EXPECT_EQ(i, 6);

    //END TEST
    if (test_session != NULL) {
        system("rm -rf /home/enricsosa/bsc/cassandra-master/arrow/test");

        CassFuture *close_future = cass_session_close(test_session);
        cass_future_free(close_future);
        cass_session_free(test_session);
        cass_cluster_free(test_cluster);
        test_session = NULL;
    }
}
