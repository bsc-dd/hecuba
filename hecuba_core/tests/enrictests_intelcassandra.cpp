#include <iostream>
#include <cassandra.h>
#include "gtest/gtest.h"
#include "../src/CacheTable.h"
#include "../src/StorageInterface.h"

#include <arrow/memory_pool.h>
#include <arrow/status.h>
#include <arrow/ipc/reader.h>
#include <arrow/buffer.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/io/api.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/interfaces.h>
#include <arrow/ipc/feather.h>
#include <arrow/array/builder_binary.h>

#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

using namespace std;

#define PY_ERR_CHECK if (PyErr_Occurred()){PyErr_Print(); PyErr_Clear();}
#define PMEM_OFFSET 8

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

TEST(IntelCassandra, IntelCassandra_test1) {

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

    arrow::Status status;
    auto memory_pool = arrow::default_memory_pool(); //arrow
    auto field = arrow::field("field", arrow::binary());
    std::vector<std::shared_ptr<arrow::Field>> fields = {field};
    auto schema = std::make_shared<arrow::Schema>(fields);

    vector<string> arrowFiles(6);
    CassStatement* statement = NULL;
    const char* query = "INSERT INTO test_arrow.integers_buffer (storage_id, col_id, row_id, size_elem, payload) VALUES (?, ?, ?, ?, ?);";
    for (int i = 0; i < 6; ++i) {
        arrow::BinaryBuilder builder(arrow::binary(), memory_pool); //arrow
        status = builder.Resize(1); //arrow
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at builder.Resize" << std::endl;
        uint64_t payload = i;
        //std::unique_ptr<char[]> buffer(new char[sizeof(uint64_t)]);
        char* buffer = new char[sizeof(uint64_t)];
        memcpy(buffer, &payload, sizeof(uint64_t));
        status = builder.Append(buffer, sizeof(uint64_t));

        std::shared_ptr<arrow::Array> array;
        status = builder.Finish(&array);
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at builder.Finish" << std::endl;
        auto batch = arrow::RecordBatch::Make(schema, 1, {array});
        std::shared_ptr<arrow::io::BufferOutputStream> bufferOutputStream;
        status = arrow::io::BufferOutputStream::Create(0, memory_pool, &bufferOutputStream);
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at BufferOutputStream::Create" << std::endl;
        std::shared_ptr<arrow::ipc::RecordBatchWriter> file_writer;
        status = arrow::ipc::RecordBatchFileWriter::Open(bufferOutputStream.get(), schema, &file_writer);
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at RecordBatchFileWriter::Open" << std::endl;

        status = file_writer->WriteRecordBatch(*batch);
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at file_writer->WriteRecordBatch" << std::endl;
        status = file_writer->Close();
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at file_writer->Close" << std::endl;
        status = bufferOutputStream->Close();
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at bufferOutputStream->Close" << std::endl;

        std::shared_ptr<arrow::Buffer> result;
        status = bufferOutputStream->Finish(&result); //arrow
        arrowFiles[i] = result->ToString();
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at bufferOutputStream->Finish" << std::endl;

        statement = cass_statement_new(query, 5);
        CassUuid uuid = {};
        cass_uuid_from_string("c37d661d-7e61-49ea-96a5-68c34e83db3a", &uuid);
        cass_statement_bind_uuid(statement, 0, uuid);
        cass_statement_bind_int64(statement, 1, (cass_int64_t)i);
        cass_statement_bind_int64(statement, 2, (cass_int64_t)1);
        cass_statement_bind_int32(statement, 3, (cass_int32_t)8);
        cass_statement_bind_bytes(statement, 4, (cass_byte_t*)result->data(), result->size());

        fireandforgetstatement(statement, test_session);
    }

    const CassResult *result = fireandgetresult("SELECT * FROM test_arrow.integers_buffer;", test_session);
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

    std::shared_ptr<arrow::ipc::RecordBatchFileReader> sptrFileReader;
    //open devdax access
    int fdIn = open("/home/enricsosa/bsc/cassandra/arrow_persistent_heap", O_RDONLY);  //TODO handle open errors; define file name
    if (fdIn < 0) {
        throw ModuleException("open error");
    }

    i = 0;
    while (cass_iterator_next(rows)) {
        const CassRow* row = cass_iterator_get_row(rows);
        const CassValue* cassValueAddr = cass_row_get_column_by_name(row, "arrow_addr");
        const CassValue* cassValueSize = cass_row_get_column_by_name(row, "arrow_size");
        //read row: we get where it is allocated in devdax
        cass_int64_t addr;
        cass_int32_t size;
        cass_value_get_int64(cassValueAddr, &addr);
        cass_value_get_int32(cassValueSize, &size);

        off_t dd_addr = addr+PMEM_OFFSET;
        off_t page_addr = dd_addr & ~(sysconf(_SC_PAGE_SIZE) - 1); //I don't even remember how this works

        //read from devdax
        off_t page_offset = dd_addr-page_addr;
        //allocates in memory [...]
        size_t total_arrow_size = size+page_offset;
        unsigned char* src = (unsigned char*) mmap(NULL, total_arrow_size, PROT_READ, MAP_SHARED, fdIn, page_addr); //TODO handle mmap errors
        if (src == MAP_FAILED) {
            throw ModuleException("mmap error");
        }

        //read from devdax
        std::string fileAsString(reinterpret_cast<char*>(&src[page_offset]), size);
        EXPECT_EQ(fileAsString, arrowFiles[i]);
        arrow::io::BufferReader bufferReader(fileAsString);

        arrow::Status status;
        status = arrow::ipc::RecordBatchFileReader::Open(&bufferReader, &sptrFileReader); //TODO handle RecordBatchFileReader::Open errors
        if (not status.ok()) {
            throw ModuleException("RecordBatchFileReader::Open error");
        }

        const std::shared_ptr<arrow::ipc::RecordBatchFileReader> localPtr = sptrFileReader;
        int num_batches = localPtr->num_record_batches();

        for (int i = 0; i < num_batches; ++i) { //for each batch inside arrow File; Theoretically, there should be one batch
            std::shared_ptr<arrow::RecordBatch> chunk;
            status = localPtr->ReadRecordBatch(i, &chunk);
            if (not status.ok()) { //TODO ReadRecordBatch
                throw ModuleException("ReadRecordBatch error");
            }
            std::shared_ptr<arrow::Array> col = chunk->column(0); //Theoretically, there must be one column

            std::shared_ptr<arrow::BinaryArray> data = std::dynamic_pointer_cast<arrow::BinaryArray>(col);

            int length;
            for (int k = 0; k < col->length(); ++k) {
                const uint8_t* bytes = data->GetValue(k, &length);
            }
        }
        ++i;
    }
    EXPECT_EQ(i, 6);

    //END TEST
    if (test_session != NULL) {
        sptrFileReader.reset();
        close(fdIn);

        CassFuture *close_future = cass_session_close(test_session);
        cass_future_free(close_future);
        cass_session_free(test_session);
        cass_cluster_free(test_cluster);
        test_session = NULL;
    }
}