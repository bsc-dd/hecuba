#include <iostream>
#include <cassandra.h>
#include "gtest/gtest.h"
#include "../src/MetaManager.h"
#include "../src/StorageInterface.h"


CassSession *test_session = nullptr;
CassCluster *test_cluster = nullptr;

const std::string table = "hfetch_test";
const std::string keyspace = "test";

// CQL statements
const std::string create_ksp = "CREATE KEYSPACE " + keyspace +
                               +" WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};";
const std::string drop_ksp = "DROP KEYSPACE IF EXISTS " + keyspace + ";";
const std::string create_table = "CREATE TABLE " + keyspace + "." + table +
                                 +"(storage_id uuid PRIMARY KEY, class_name text, name text);";

/** TEST SETUP **/

void fireandforget(const std::string &query) {
    std::cout << "EXECUTING " << query << std::endl;
    CassStatement *statement = cass_statement_new(query.c_str(), 0);
    CassFuture *connect_future = cass_session_execute(test_session, statement);
    CassError rc = cass_future_error_code(connect_future);
    if (rc != CASS_OK) {
        std::cout << "ERROR ON EXECUTING QUERY: " << cass_error_desc(rc) << std::endl;
    }
    EXPECT_TRUE(rc == CASS_OK);
    cass_future_free(connect_future);
    cass_statement_free(statement);
}


void setupcassandra(std::string &contact_points, uint64_t nodePort) {

    CassFuture *connect_future = nullptr;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_points.c_str());
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect(test_session, test_cluster);
    CassError rc = cass_future_error_code(connect_future);
    cass_future_free(connect_future);

    EXPECT_TRUE(rc == CASS_OK);

    fireandforget(drop_ksp);
    fireandforget(create_ksp);
    fireandforget(create_table);
}

void teardown() {
    if (test_session != nullptr) {
        CassFuture *close_future = cass_session_close(test_session);
        cass_future_free(close_future);
        cass_session_free(test_session);
        cass_cluster_free(test_cluster);
        test_session = nullptr;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::string contact_p = "127.0.0.1";
    uint32_t nodePort = 9042;

    setupcassandra(contact_p, nodePort);

    int ret_code = RUN_ALL_TESTS();

    teardown();

    return ret_code;
}


TEST(TestMetaManager, RegisterObj) {
    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "storage_id"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "name"}}};

    std::map<std::string, std::string> config;
    config["writer_par"] = "1";
    config["writer_buffer"] = "1";
    config["timestamped_writes"] = "false";


    TableMetadata *table_meta = new TableMetadata(table.c_str(), keyspace.c_str(), keysnames, colsnames, test_session);

    MetaManager *meta_manager = new MetaManager(table_meta, test_session, config);


    uint64_t *storage_id = new uint64_t[2]{4984489798, 78587497499};
    CassUuid cass_uuid = {storage_id[0], storage_id[1]};

    std::string obj_name = "obj_name_0";
    ArrayMetadata *metas = nullptr;

    meta_manager->register_obj(storage_id, obj_name, metas);

    std::string istorage_select = "SELECT name FROM " + keyspace + "." + table + " WHERE storage_id=?";
    CassStatement *statement = cass_statement_new(istorage_select.c_str(), 1);

    cass_statement_bind_uuid(statement, 0, cass_uuid);
    CassFuture *connect_future = cass_session_execute(test_session, statement);
    CassError rc = cass_future_error_code(connect_future);

    if (rc != CASS_OK) {
        std::cerr << "Meta Manager failure" << std::endl;
        std::cerr << std::string(cass_error_desc(rc)) << std::endl;
        std::cerr << istorage_select << std::endl;

    }

    EXPECT_TRUE(rc == CASS_OK);
    cass_future_free(connect_future);
    cass_statement_free(statement);

    delete (meta_manager);
    delete (table_meta);
}


TEST(TestMetaManager, StorageInterfaceMake) {
    std::string contact_p = "127.0.0.1";
    uint32_t nodePort = 9042;

    StorageInterface *SI = new StorageInterface(nodePort, contact_p);

    EXPECT_NE(SI, nullptr);


    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "storage_id"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "name"}}};

    std::map<std::string, std::string> config;
    config["writer_par"] = "1";
    config["writer_buffer"] = "1";
    config["timestamped_writes"] = "false";


    MetaManager *meta_manager = SI->make_meta_manager(table.c_str(), keyspace.c_str(), keysnames, colsnames, config);


    uint64_t *storage_id = new uint64_t[2]{123456789123456789, 123456789123456789};
    CassUuid cass_uuid = {storage_id[0], storage_id[1]};

    std::string obj_name = "obj_name_2";
    ArrayMetadata *metas = nullptr;

    meta_manager->register_obj(storage_id, obj_name, metas);

    std::string istorage_select = "SELECT name FROM " + keyspace + "." + table + " WHERE storage_id=?";
    CassStatement *statement = cass_statement_new(istorage_select.c_str(), 1);

    cass_statement_bind_uuid(statement, 0, cass_uuid);
    CassFuture *connect_future = cass_session_execute(test_session, statement);
    CassError rc = cass_future_error_code(connect_future);

    if (rc != CASS_OK) {
        std::cerr << "Meta Manager failure" << std::endl;
        std::cerr << std::string(cass_error_desc(rc)) << std::endl;
        std::cerr << istorage_select << std::endl;

    }

    EXPECT_TRUE(rc == CASS_OK);
    cass_future_free(connect_future);
    cass_statement_free(statement);

    delete (meta_manager);
    delete (SI);
}