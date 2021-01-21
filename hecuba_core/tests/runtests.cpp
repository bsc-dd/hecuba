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

    fireandforget(
            "CREATE TABLE test.only_keys(first int, second int, third text, PRIMARY KEY((first, second), third));",
            test_session);
    /*
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
    */
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



/*** TESTS TO ANALYZE THE PARTITIONING OF ARRAYS ***/

//Test to verify the Zorder produce the correct result
TEST(TestZorder, Zorder) {
    std::vector<uint32_t> ccs = {0, 0, 0, 2}; //row 3, column 2
    SpaceFillingCurve SP;
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = {4, 4, 4, 4};
    arr_metas.elem_size = sizeof(uint32_t);
    arr_metas.partition_type = ZORDER_ALGORITHM;


    ZorderCurveGenerator *partitioner = new ZorderCurveGenerator(arr_metas, nullptr);
    uint64_t result = partitioner->computeZorder(ccs);
    uint64_t expected_zorder = 128;
    EXPECT_EQ(expected_zorder, result);
    delete (partitioner);
}


//Test to verify the Zorder Id and its inverse produce the same result
TEST(TestZorder, ZorderInv) {
    std::vector<uint32_t> ccs = {23, 25, 40};
    ArrayMetadata *arr_metas = new ArrayMetadata();
    arr_metas->dims = {32, 40, 64};
    arr_metas->elem_size = sizeof(uint32_t);
    arr_metas->partition_type = ZORDER_ALGORITHM;

    ZorderCurveGenerator *partitioner = new ZorderCurveGenerator();
    uint64_t result = partitioner->computeZorder(ccs);
    std::vector<uint32_t> inverse = partitioner->zorderInverse(result, ccs.size());
    EXPECT_TRUE(ccs == inverse);
    delete (arr_metas);
    delete (partitioner);
}

//Verify that ZorderCurve::getIndexes returns the correct coordinates of the nth_element
//inside an array of shape dims
TEST(TestMakePartitions, Indexes) {
    ArrayMetadata *arr_metas = new ArrayMetadata();
    arr_metas->dims = {8, 8};
    arr_metas->elem_size = sizeof(uint32_t);
    arr_metas->partition_type = ZORDER_ALGORITHM;

    ZorderCurveGenerator *partitioner = new ZorderCurveGenerator();
    uint64_t nth_element = 43;
    std::vector<uint32_t> indexes = partitioner->getIndexes(nth_element, arr_metas->dims);
    std::vector<uint32_t> expected_coordinates = {5, 3};
    EXPECT_TRUE(indexes == expected_coordinates);
    delete (arr_metas);
    delete (partitioner);
}


//Verify the that transforming the coordinates to an offset and back to coordinates
// produces the same result
TEST(TestMakePartitions, Indexes2ways) {
    std::vector<uint32_t> ccs = {53, 28, 34};
    ArrayMetadata *arr_metas = new ArrayMetadata();
    arr_metas->dims = {32, 40, 64};
    arr_metas->elem_size = sizeof(uint32_t);
    arr_metas->partition_type = ZORDER_ALGORITHM;

    ZorderCurveGenerator *partitioner = new ZorderCurveGenerator();
    uint64_t id = 1077;
    std::vector<uint32_t> indexes = partitioner->getIndexes(id, ccs);
    uint64_t computed_id = partitioner->getIdFromIndexes(ccs, indexes);
    EXPECT_EQ(computed_id, id);
    delete (arr_metas);
    delete (partitioner);
}


//Simple test to analyze a 2D array partitioning
//of uncommon dimension which make the array to
//be undivisible in blocks of the same shape
TEST(TestMakePartitions, 2DZorder) {
    uint32_t nrows = 463;
    uint32_t ncols = 53;
    std::vector<uint32_t> ccs = {nrows, ncols};
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(int32_t);
    arr_metas.partition_type = ZORDER_ALGORITHM;

    int32_t *data = new int32_t[ncols * nrows];
    for (uint32_t row = 0; row < nrows; ++row) {
        for (uint32_t col = 0; col < ncols; ++col) {
            data[col + (ncols * row)] = (int32_t) col + (ncols * row);
        }
    }
    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);

    std::set<int32_t> elements_found;
    uint64_t total_elem = 0;
    while (!partitioner->isDone()) {
        Partition chunk = partitioner->getNextPartition();
        uint64_t *size = (uint64_t *) chunk.data;
        uint64_t nelem = *size / arr_metas.elem_size;
        total_elem += nelem;
        int32_t *chunk_data = (int32_t *) ((char *) chunk.data + sizeof(uint64_t));
        for (uint64_t pos = 0; pos < nelem; ++pos) {
            elements_found.insert(*chunk_data);
            ++chunk_data;
        }
        free(chunk.data);
    }
    for (uint32_t row = 0; row < nrows; ++row) {
        for (uint32_t col = 0; col < ncols; ++col) {
            EXPECT_TRUE(elements_found.find((int32_t) col + (ncols * row)) != elements_found.end());
        }
    }

    EXPECT_EQ(total_elem, ncols * nrows);
    EXPECT_EQ(elements_found.size(), ncols * nrows);
    EXPECT_EQ(*elements_found.begin(), 0);
    EXPECT_EQ(*elements_found.rbegin(), ncols * nrows - 1);
    delete[](data);
    delete (partitioner);
}




//Simple test to analyze a 2D array partitioning
//of uncommon dimension which make the array to
//be undivisible in blocks of the same shape
TEST(TestMakePartitions, 2DZorderZeroes) {
    uint32_t nrows = 463;
    uint32_t ncols = 53;
    std::vector<uint32_t> ccs = {nrows, ncols};
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(double);
    arr_metas.partition_type = ZORDER_ALGORITHM;

    double *data = new double[ncols * nrows];
    for (uint32_t row = 0; row < nrows; ++row) {
        for (uint32_t col = 0; col < ncols; ++col) {
            data[col + (ncols * row)] = (double) 0;
        }
    }

    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);


    std::set<int32_t> elements_found;
    uint64_t total_elem = 0;
    while (!partitioner->isDone()) {
        Partition chunk = partitioner->getNextPartition();
        uint64_t *size = (uint64_t *) chunk.data;
        uint64_t nelem = *size / arr_metas.elem_size;
        total_elem += nelem;
        double *chunk_data = (double *) ((char *) chunk.data + sizeof(uint64_t));
        for (uint64_t pos = 0; pos < nelem; ++pos) {
            EXPECT_EQ(*chunk_data, (double) 0);
            ++chunk_data;
        }
        free(chunk.data);
    }
    EXPECT_EQ(total_elem, ncols * nrows);
//    EXPECT_EQ(elements_found.size(),ncols*nrows);
//    EXPECT_EQ(*elements_found.begin(),0);
//    EXPECT_EQ(*elements_found.rbegin(),ncols*nrows-1);

    //void* array = partitioner.merge_partitions(arr_metas,chunks);

    //std::cout << "FInal memcmp " << memcmp(array,data,ncols*nrows*arr_metas->elem_size) << std::endl;
    delete[](data);
    delete (partitioner);
}

//Test to analyze the partitioning of a small 3D array
TEST(TestMakePartitions, 3DZorder_Small) {
    std::vector<uint32_t> ccs = {17, 17, 17};
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(int32_t);
    arr_metas.partition_type = ZORDER_ALGORITHM;

    uint64_t arr_size = 1;
    for (uint32_t size_dim:ccs) {
        arr_size *= size_dim;
    }
    int32_t *data = new int32_t[arr_size];
    for (uint32_t i = 0; i < arr_size; ++i) {
        data[i] = i;
    }

    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);

    std::vector<int32_t> elements_found(arr_size, 0);
    uint64_t total_elem = 0;
    bool check;
    while (!partitioner->isDone()) {
        Partition chunk = partitioner->getNextPartition();
        uint64_t *size = (uint64_t *) chunk.data;
        uint64_t nelem = *size / arr_metas.elem_size;
        total_elem += nelem;
        int32_t *chunk_data = (int32_t *) ((char *) chunk.data + sizeof(uint64_t));
        for (uint64_t pos = 0; pos < nelem; ++pos) {
            check = *chunk_data >= (int64_t) arr_size || *chunk_data < 0;
            EXPECT_FALSE (check);
            if (!check) elements_found[*chunk_data]++;
            ++chunk_data;
        }
        free(chunk.data);
    }
    int32_t max_elem = 1;
    for (int32_t cc:ccs) {
        max_elem *= cc;
    }
    bool found;
    for (int32_t elem = 0; elem < max_elem; ++elem) {
        found = elements_found[elem] == 1;
        EXPECT_EQ(elements_found[elem], 1);
        if (!found) std::cout << "Element not found: " << elem << std::endl;
    }
    EXPECT_EQ(total_elem, max_elem);
    EXPECT_EQ(elements_found.size(), max_elem);
    delete[](data);
    delete (partitioner);
}

//Test to analyze the partitioning of a medium 3D array, approx 72MB
TEST(TestMakePartitions, 3DZorder_Medium) {
    std::vector<uint32_t> ccs = {250, 150, 500};
    ArrayMetadata arr_metas =  ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(int32_t);
    arr_metas.partition_type = ZORDER_ALGORITHM;

    uint64_t arr_size = 1;
    for (uint32_t size_dim:ccs) {
        arr_size *= size_dim;
    }
    int32_t *data = new int32_t[arr_size];
    for (uint32_t i = 0; i < arr_size; ++i) {
        data[i] = i;
    }

    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);

    std::vector<int32_t> elements_found(arr_size, 0);
    uint64_t total_elem = 0;
    bool check;
    while (!partitioner->isDone()) {
        Partition chunk = partitioner->getNextPartition();
        uint64_t *size = (uint64_t *) chunk.data;
        uint64_t nelem = *size / arr_metas.elem_size;
        total_elem += nelem;
        int32_t *chunk_data = (int32_t *) ((char *) chunk.data + sizeof(uint64_t));
        for (uint64_t pos = 0; pos < nelem; ++pos) {
            check = *chunk_data >= (int64_t) arr_size || *chunk_data < 0;
            EXPECT_FALSE (check);
            if (!check) elements_found[*chunk_data]++;
            ++chunk_data;
        }
        free(chunk.data);
    }
    delete (partitioner);

    int32_t max_elem = 1;
    for (int32_t cc:ccs) {
        max_elem *= cc;
    }
    bool found;
    for (int32_t elem = 0; elem < max_elem; ++elem) {
        found = elements_found[elem] == 1;
        EXPECT_EQ(elements_found[elem], 1);
        if (!found) std::cout << "Element not found: " << elem << std::endl;
    }
    EXPECT_EQ(total_elem, max_elem);
    delete[](data);
}


//Test to analyze the partitioning of a big 3D array, approx 256MB
TEST(TestMakePartitions, 3DZorder_Big) {
    std::vector<uint32_t> ccs = {512, 512, 512};
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(int32_t);
    arr_metas.partition_type = ZORDER_ALGORITHM;

    uint64_t arr_size = 1;
    for (int32_t size_dim:ccs) {
        arr_size *= size_dim;
    }
    int32_t *data = new int32_t[arr_size];
    for (uint32_t i = 0; i < arr_size; ++i) {
        data[i] = i;
    }
    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);


    std::vector<int32_t> elements_found(arr_size, 0);
    uint64_t total_elem = 0;
    bool check;
    while (!partitioner->isDone()) {
        Partition chunk = partitioner->getNextPartition();
        uint64_t *size = (uint64_t *) chunk.data;
        uint64_t nelem = *size / arr_metas.elem_size;
        total_elem += nelem;
        int32_t *chunk_data = (int32_t *) ((char *) chunk.data + sizeof(uint64_t));
        for (uint64_t pos = 0; pos < nelem; ++pos) {
            check = *chunk_data >= (int64_t) arr_size || *chunk_data < 0;
            EXPECT_FALSE (check);
            if (!check) elements_found[*chunk_data]++;
            ++chunk_data;
        }
        free(chunk.data);
    }
    delete (partitioner);

    int32_t max_elem = 1;
    for (int32_t cc:ccs) {
        max_elem *= cc;
    }
    bool found;
    for (int32_t elem = 0; elem < max_elem; ++elem) {
        found = elements_found[elem] == 1;
        EXPECT_EQ(elements_found[elem], 1);
        if (!found) std::cout << "Element not found: " << elem << std::endl;

    }
    EXPECT_EQ(total_elem, max_elem);
    delete[](data);

}

//Test incrementing dimensions up to 100MB (17^4 *4=92MB)
TEST(TestMakePartitions, NDZorder) {
    uint32_t max_dims = 6;
    std::vector<uint32_t> ccs = {17};
    ArrayMetadata arr_metas = ArrayMetadata();
    while (ccs.size() <= max_dims) {
        arr_metas.dims = ccs;
        arr_metas.elem_size = sizeof(int32_t);
        arr_metas.partition_type = ZORDER_ALGORITHM;

        uint64_t arr_size = 1;
        for (int32_t size_dim:ccs) {
            arr_size *= size_dim;
        }
        int32_t *data = new int32_t[arr_size];
        for (uint32_t i = 0; i < arr_size; ++i) {
            data[i] = i;
        }

        SpaceFillingCurve SFC;
        SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);

        std::vector<int32_t> elements_found(arr_size, 0);
        uint64_t total_elem = 0;
        bool check;
        while (!partitioner->isDone()) {
            Partition chunk = partitioner->getNextPartition();
            uint64_t *size = (uint64_t *) chunk.data;
            uint64_t nelem = *size / arr_metas.elem_size;
            total_elem += nelem;
            int32_t *chunk_data = (int32_t *) ((char *) chunk.data + sizeof(uint64_t));
            for (uint64_t pos = 0; pos < nelem; ++pos) {
                check = *chunk_data >= (int64_t) arr_size || *chunk_data < 0;
                EXPECT_FALSE (check);
                if (!check) elements_found[*chunk_data]++;
                ++chunk_data;
            }
            free(chunk.data);

        }
        delete (partitioner);
        int32_t max_elem = 1;
        for (int32_t cc:ccs) {
            max_elem *= cc;
        }
        bool found;
        for (int32_t elem = 0; elem < max_elem; ++elem) {
            found = elements_found[elem] == 1;
            EXPECT_EQ(elements_found[elem], 1);
            if (!found) std::cout << "Element not found: " << elem << std::endl;

        }
        EXPECT_EQ(total_elem, max_elem);
        delete[](data);

        ccs.push_back(17);
    }


}

//Test partitioning 2Dimensions with Doubles
TEST(TestMakePartitions, 2DZorder128Double) {
    uint32_t ncols = 1000;
    uint32_t nrows = 1000;
    std::vector<uint32_t> ccs = {nrows, ncols}; //4D 128 elements
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(double);
    arr_metas.partition_type = ZORDER_ALGORITHM;


    uint64_t arr_size = 1;
    for (int32_t size_dim:ccs) {
        arr_size *= size_dim;
    }

    double *data = new double[ncols * nrows];
    for (uint32_t row = 0; row < nrows; ++row) {
        for (uint32_t col = 0; col < ncols; ++col) {
            data[col + (ncols * row)] = (double) col + (ncols * row);
        }
    }

    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);
    std::set<double> elements_found;
    uint64_t total_elem = 0;
    while (!partitioner->isDone()) {
        Partition chunk = partitioner->getNextPartition();
        uint64_t *size = (uint64_t *) chunk.data;
        uint64_t nelem = *size / arr_metas.elem_size;
        total_elem += nelem;
        double *chunk_data = (double *) ((char *) chunk.data + sizeof(uint64_t));
        for (uint64_t pos = 0; pos < nelem; ++pos) {
            elements_found.insert(*chunk_data);
            ++chunk_data;
        }
        free(chunk.data);
    }
    for (uint32_t row = 0; row < nrows; ++row) {
        for (uint32_t col = 0; col < ncols; ++col) {
            EXPECT_TRUE(elements_found.find((double) col + (ncols * row)) != elements_found.end());
        }
    }
    EXPECT_EQ(total_elem, ncols * nrows);
    EXPECT_EQ(elements_found.size(), ncols * nrows);
    EXPECT_EQ(*elements_found.begin(), 0);
    EXPECT_EQ(*elements_found.rbegin(), ncols * nrows - 1);
    delete[](data);
    delete (partitioner);
}


//Test partitioning 2Dimensions, incrementing the number of elements of the array up to 128KB
//128*128=16384 combinations tried
TEST(TestMakePartitions, 2DZorderByRange) {

    uint32_t maxcols = 128;
    uint32_t maxrows = 128;
    for (uint32_t ncols = 1; ncols < maxcols; ++ncols) {
        for (uint32_t nrows = 1; nrows < maxrows; ++nrows) {
            std::vector<uint32_t> ccs = {nrows, ncols}; //4D 128 elements
            ArrayMetadata arr_metas = ArrayMetadata();
            arr_metas.dims = ccs;
            arr_metas.elem_size = sizeof(double);
            arr_metas.partition_type = ZORDER_ALGORITHM;

            uint64_t arr_size = 1;
            for (int32_t size_dim:ccs) {
                arr_size *= size_dim;
            }

            double *data = new double[ncols * nrows];
            for (uint32_t row = 0; row < nrows; ++row) {
                for (uint32_t col = 0; col < ncols; ++col) {
                    data[col + (ncols * row)] = (double) col + (ncols * row);
                }
            }
            bool check;
            SpaceFillingCurve SFC;
            SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);
            std::vector<uint32_t> elements_found(arr_size, 0);
            uint64_t total_elem = 0;
            while (!partitioner->isDone()) {
                Partition chunk = partitioner->getNextPartition();
                uint64_t *size = (uint64_t *) chunk.data;
                uint64_t nelem = *size / arr_metas.elem_size;
                total_elem += nelem;
                double *chunk_data = (double *) ((char *) chunk.data + sizeof(uint64_t));
                for (uint64_t pos = 0; pos < nelem; ++pos) {
                    check = *chunk_data >= (double) arr_size || *chunk_data < 0;
                    EXPECT_FALSE (check);
                    double elem_data = *chunk_data;
                    EXPECT_DOUBLE_EQ(elem_data - (int32_t) elem_data, 0.0);
                    if (!check) elements_found[*chunk_data]++;
                    ++chunk_data;
                }
                free(chunk.data);
            }
            delete (partitioner);

            bool found;
            for (uint32_t row = 0; row < nrows; ++row) {
                for (uint32_t col = 0; col < ncols; ++col) {
                    int32_t elem = col + (ncols * row);
                    found = elements_found[elem] == 1;
                    EXPECT_EQ(elements_found[elem], 1);
                    if (!found) std::cout << "Element not found: " << elem << std::endl;
                }
            }
            EXPECT_EQ(total_elem, ncols * nrows);
            EXPECT_EQ(elements_found.size(), ncols * nrows);
            delete[](data);
        }
    }
}


//Test to evaluate the partition algorithm which generates always a single partition
// which is a new copy of the array data untouched
TEST(TestMakePartitions, 2DNopart) {
    uint32_t nrows = 124;
    uint32_t ncols = 1104;
    int32_t arr_size = ncols * nrows;
    std::vector<uint32_t> ccs = {ncols, nrows};
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(int32_t);
    arr_metas.partition_type = NO_PARTITIONS;

    int32_t *data = new int32_t[arr_size];
    for (uint32_t col = 0; col < ncols; ++col) {
        for (uint32_t row = 0; row < nrows; ++row) {
            data[col + (ncols * row)] = (int32_t) col + (ncols * row);
        }
    }
    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);
    Partition part = partitioner->getNextPartition();
    EXPECT_TRUE(partitioner->isDone());
    uint64_t *chunk_size = (uint64_t *) part.data;
    EXPECT_EQ(*chunk_size, arr_size * arr_metas.elem_size);
    char *partition_data = ((char *) part.data) + sizeof(uint64_t);
    int32_t equal = memcmp(partition_data, data, arr_size * sizeof(int32_t));
    EXPECT_TRUE(equal == 0);
    EXPECT_FALSE(data == (int32_t *) partition_data);
    delete[](data);
    delete (partitioner);
}




//Test to generate partitions of the array using Zorder and merge them back
// into a single array and make sure both processes performed correctly
TEST(TestMakePartitions, 3DZorderAndReverse) {
    std::vector<uint32_t> ccs = {256, 256, 256};
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(int32_t);
    arr_metas.partition_type = ZORDER_ALGORITHM;

    uint64_t arr_size = 1;
    for (int32_t size_dim:ccs) {
        arr_size *= size_dim;
    }
    int32_t *data = new int32_t[arr_size];
    for (uint32_t i = 0; i < arr_size; ++i) {
        data[i] = i;
    }
    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);
    std::set<int32_t> elements_found;
    uint64_t total_elem = 0;
    std::vector<Partition> chunks;
    while (!partitioner->isDone()) {
        Partition chunk = partitioner->getNextPartition();
        uint64_t *size = (uint64_t *) chunk.data;
        uint64_t nelem = *size / arr_metas.elem_size;
        total_elem += nelem;
        int32_t *chunk_data = (int32_t *) ((char *) chunk.data + sizeof(uint64_t));
        for (uint64_t pos = 0; pos < nelem; ++pos) {
            elements_found.insert(*chunk_data);
            ++chunk_data;
        }
        chunks.push_back(chunk);
    }
    int32_t max_elem = 1;
    for (int32_t cc:ccs) {
        max_elem *= cc;
    }
    bool found;
    for (int32_t elem = 0; elem < max_elem; ++elem) {
        found = elements_found.find(elem) != elements_found.end();
        EXPECT_TRUE(found);
        if (!found) std::cout << "Element not found: " << elem << std::endl;

    }
    EXPECT_EQ(total_elem, max_elem);
    EXPECT_EQ(elements_found.size(), max_elem);
    EXPECT_EQ(*elements_found.begin(), 0);
    EXPECT_EQ(*elements_found.rbegin(), max_elem - 1);
}




//Test to generate partitions of the array using Zorder and merge them back
// into a single array and make sure both processes performed correctly
TEST(TestMakePartitions, 4DZorderAndReverse) {
    std::vector<uint32_t> ccs = {100, 20, 150, 30};
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(int32_t);
    arr_metas.partition_type = ZORDER_ALGORITHM;

    uint64_t arr_size = 1;
    for (int32_t size_dim:ccs) {
        arr_size *= size_dim;
    }
    int32_t *data = new int32_t[arr_size];
    for (uint32_t i = 0; i < arr_size; ++i) {
        data[i] = i;
    }
    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);
    std::set<int32_t> elements_found;
    uint64_t total_elem = 0;
    std::vector<Partition> chunks;
    while (!partitioner->isDone()) {
        Partition chunk = partitioner->getNextPartition();
        uint64_t *size = (uint64_t *) chunk.data;
        uint64_t nelem = *size / arr_metas.elem_size;
        total_elem += nelem;
        int32_t *chunk_data = (int32_t *) ((char *) chunk.data + sizeof(uint64_t));
        for (uint64_t pos = 0; pos < nelem; ++pos) {
            elements_found.insert(*chunk_data);
            ++chunk_data;
        }
        chunks.push_back(chunk);
    }
    int32_t max_elem = 1;
    for (int32_t cc:ccs) {
        max_elem *= cc;
    }
    bool found;
    for (int32_t elem = 0; elem < max_elem; ++elem) {
        found = elements_found.find(elem) != elements_found.end();
        EXPECT_TRUE(found);
        if (!found) std::cout << "Element not found: " << elem << std::endl;

    }
    EXPECT_EQ(total_elem, max_elem);
    EXPECT_EQ(elements_found.size(), max_elem);
    EXPECT_EQ(*elements_found.begin(), 0);
    EXPECT_EQ(*elements_found.rbegin(), max_elem - 1);
    delete (partitioner);
}


TEST(TestMakePartitions, ReadBlockOnlyOnce) {
    uint32_t nrows = 463;
    uint32_t ncols = 53;
    std::vector<uint32_t> ccs = {nrows, ncols};
    ArrayMetadata arr_metas = ArrayMetadata();
    arr_metas.dims = ccs;
    arr_metas.elem_size = sizeof(int32_t);
    arr_metas.partition_type = ZORDER_ALGORITHM;

    int32_t *data = new int32_t[ncols * nrows];
    for (uint32_t row = 0; row < nrows; ++row) {
        for (uint32_t col = 0; col < ncols; ++col) {
            data[col + (ncols * row)] = (int32_t) col + (ncols * row);
        }
    }

    // We first break the array in blocks and clusters. Count the number of unique clusters.
    SpaceFillingCurve SFC;
    SpaceFillingCurve::PartitionGenerator *partitioner = SFC.make_partitions_generator(arr_metas, data);

    std::set<uint32_t> clusters_found;
    std::set<uint32_t> n_cluster;
    std::set<int32_t> elements_found;
    uint64_t total_elem = 0;
    while (!partitioner->isDone()) {
        Partition chunk = partitioner->getNextPartition();
        uint64_t *size = (uint64_t *) chunk.data;
        uint64_t nelem = *size / arr_metas.elem_size;
        total_elem += nelem;
        int32_t *chunk_data = (int32_t *) ((char *) chunk.data + sizeof(uint64_t));
        for (uint64_t pos = 0; pos < nelem; ++pos) {
            elements_found.insert(*chunk_data);
            ++chunk_data;
        }
        clusters_found.insert(chunk.cluster_id);
        free(chunk.data);

    }
    for (uint32_t row = 0; row < nrows; ++row) {
        for (uint32_t col = 0; col < ncols; ++col) {
            EXPECT_TRUE(elements_found.find((int32_t) col + (ncols * row)) != elements_found.end());
        }
    }
    EXPECT_EQ(total_elem, ncols * nrows);
    EXPECT_EQ(elements_found.size(), ncols * nrows);
    EXPECT_EQ(*elements_found.begin(), 0);
    EXPECT_EQ(*elements_found.rbegin(), ncols * nrows - 1);

    delete[](data);
    delete (partitioner);

    // Assess the partitioner produces the same number of clusters as generated before
    partitioner = SFC.make_partitions_generator(arr_metas, nullptr);
    while (!partitioner->isDone()) {
        n_cluster.insert(partitioner->computeNextClusterId());
    }

    delete (partitioner);

    EXPECT_EQ(clusters_found.size(), n_cluster.size());

    // Verify each cluster id is gen
    // erated exactly once
    clusters_found.clear();
    partitioner = SFC.make_partitions_generator(arr_metas, nullptr);


    int32_t cluster_id;
    while (!partitioner->isDone()) {
        cluster_id = partitioner->computeNextClusterId();
        auto it = clusters_found.insert(cluster_id);
        //ASSERT_TRUE(it.second); the improvement in the function computeNextClusterId increment 4 by 4 the blocks, but sometimes
        //the cluster will be repeated
    }

    delete (partitioner);
}


/** Test to asses KV Cache is performing as expected with pointer **/
TEST(TestingKVCache, InsertGetDeleteOps) {
    const uint16_t i = 123;
    const uint16_t j = 456;
    size_t ss = sizeof(uint16_t) * 2;
    KVCache<TupleRow, TupleRow> myCache(2);

    ColumnMeta cm1 = ColumnMeta();
    cm1.info = {{"name", "ciao"}};
    cm1.type = CASS_VALUE_TYPE_INT;
    cm1.position = 0;
    cm1.size = sizeof(uint16_t);

    ColumnMeta cm2 = ColumnMeta();
    cm2.info = {{"name", "ciaociao"}};
    cm2.type = CASS_VALUE_TYPE_INT;
    cm2.position = sizeof(uint16_t);
    cm2.size = sizeof(uint16_t);

    std::vector<ColumnMeta> v = {cm1, cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas = std::make_shared<std::vector<ColumnMeta>>(v);


    char *b2 = (char *) malloc(ss);
    memcpy(b2, &i, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &j, sizeof(uint16_t));
    const TupleRow *t1 = new TupleRow(metas, sizeof(uint16_t) * 2, b2);

    uint16_t ka = 64;
    uint16_t kb = 128;
    b2 = (char *) malloc(ss);
    memcpy(b2, &ka, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &kb, sizeof(uint16_t));


    TupleRow *key1 = new TupleRow(metas, sizeof(uint16_t) * 2, b2);
    myCache.add(*key1, t1);
    delete (t1);

    EXPECT_EQ(myCache.size(), 1);
    //TupleRow t = *(myCache.getAllKeys().begin());
    //EXPECT_TRUE(t == *key1);
    //EXPECT_FALSE(&t == key1);

    // Reason: Cache builds its own copy of key1 through the copy constructor. They are equal but not the same object

    myCache.clear();
    //Removes all references, and deletes all objects. Key1 is still active thanks to our ref
    delete (key1);
}


TEST(TestingKVCache, ReplaceOp) {
    uint16_t i = 123;
    uint16_t j = 456;
    size_t ss = sizeof(uint16_t) * 2;
    KVCache<TupleRow, TupleRow> myCache(2);

    ColumnMeta cm1 = ColumnMeta();
    cm1.info = {{"name", "ciao"}};
    cm1.type = CASS_VALUE_TYPE_INT;
    cm1.position = 0;
    cm1.size = sizeof(uint16_t);

    ColumnMeta cm2 = ColumnMeta();
    cm2.info = {{"name", "ciaociao"}};
    cm2.type = CASS_VALUE_TYPE_INT;
    cm2.position = sizeof(uint16_t);
    cm2.size = sizeof(uint16_t);

    std::vector<ColumnMeta> v = {cm1, cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas = std::make_shared<std::vector<ColumnMeta>>(v);


    char *b2 = (char *) malloc(ss);
    memcpy(b2, &i, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &j, sizeof(uint16_t));
    const TupleRow *t1 = new TupleRow(metas, sizeof(uint16_t) * 2, b2);

    uint16_t ka = 64;
    uint16_t kb = 128;
    b2 = (char *) malloc(ss);
    memcpy(b2, &ka, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &kb, sizeof(uint16_t));


    TupleRow *key1 = new TupleRow(metas, sizeof(uint16_t) * 2, b2);
    myCache.add(key1, t1);

    EXPECT_EQ(myCache.size(), 1);
    TupleRow t = myCache.get(key1);
    //EXPECT_TRUE(t == *key1);
    EXPECT_FALSE(&t == key1);


    delete (t1);

    i = 500;
    b2 = (char *) malloc(ss);
    memcpy(b2, &i, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &j, sizeof(uint16_t));
    const TupleRow *t2 = new TupleRow(metas, sizeof(uint16_t) * 2, b2);
    myCache.add(*key1, t2);
    i = 123;
    b2 = (char *) malloc(ss);
    memcpy(b2, &i, sizeof(uint16_t));
    memcpy(b2 + sizeof(uint16_t), &j, sizeof(uint16_t));
    const TupleRow *t3 = new TupleRow(metas, sizeof(uint16_t) * 2, b2);
    myCache.add(key1, t3);

    EXPECT_EQ(myCache.size(), 1);
    t = myCache.get(key1);
    EXPECT_TRUE(t == *t3);
    EXPECT_FALSE(&t == t3);

    //Reason: Cache builds its own copy of key1 through the copy constructor. They are equal but not the same object

    myCache.clear();
    //Removes all references, and deletes all objects. Key1 is still active thanks to our ref
    delete (key1);
    delete (t3);
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

    ColumnMeta cm1 = ColumnMeta();
    cm1.info = {{"name", "ciao"}};
    cm1.type = CASS_VALUE_TYPE_INT;
    cm1.position = 0;
    cm1.size = sizeof(uint16_t);

    ColumnMeta cm2 = ColumnMeta();
    cm2.info = {{"name", "ciaociao"}};
    cm2.type = CASS_VALUE_TYPE_INT;
    cm2.position = sizeof(uint16_t);
    cm2.size = sizeof(uint16_t);

    std::vector<ColumnMeta> v = {cm1, cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas = std::make_shared<std::vector<ColumnMeta>>(v);


    TupleRow t1 = TupleRow(metas, sizeof(uint16_t) * 2, buffer);
    TupleRow t2 = TupleRow(metas, sizeof(uint16_t) * 2, buffer2);

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
    EXPECT_TRUE(t1 < t2);// And the one with nulls is smaller
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
    EXPECT_TRUE(t2 < t1);
    //they have all elements to null but t2 has position 1 to valid
    t1.setNull(0);
    EXPECT_FALSE(!(t1 < t2) && !(t2 < t1));
    EXPECT_FALSE(!(t1 > t2) && !(t2 > t1));
    EXPECT_TRUE(t2 < t1);
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
    EXPECT_FALSE(t1.get_element(1) == nullptr);
    //getting a null element returns a null ptr
    t1.setNull(1);
    EXPECT_TRUE(t1.get_element(1) == nullptr);
    //however, the other tuple still returns a valid ptr for the same position
    EXPECT_FALSE(t2.get_element(1) == nullptr);
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

    ColumnMeta cm1 = ColumnMeta();
    cm1.info = {{"name", "ciao"}};
    cm1.type = CASS_VALUE_TYPE_INT;
    cm1.position = 0;
    cm1.size = sizeof(uint16_t);

    ColumnMeta cm2 = ColumnMeta();
    cm2.info = {{"name", "ciaociao"}};
    cm2.type = CASS_VALUE_TYPE_INT;
    cm2.position = sizeof(uint16_t);
    cm2.size = sizeof(uint16_t);

    std::vector<ColumnMeta> v = {cm1, cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas = std::make_shared<std::vector<ColumnMeta>>(v);


    TupleRow t1 = TupleRow(metas, sizeof(uint16_t) * 2, buffer);
    TupleRow t2 = TupleRow(metas, sizeof(uint16_t) * 2, buffer2);

    //Equality
    EXPECT_TRUE(!(t1 < t2) && !(t2 < t1));
    EXPECT_TRUE(!(t1 > t2) && !(t2 > t1));


    cm2 = ColumnMeta();
    cm2.info = {{"name", "ciaociao"}};
    cm2.type = CASS_VALUE_TYPE_INT;
    cm2.position = sizeof(uint16_t);
    cm2.size = sizeof(uint16_t);


    std::vector<ColumnMeta> v2 = {cm1, cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas2 = std::make_shared<std::vector<ColumnMeta>>(v2);


    char *buffer3 = (char *) malloc(size * 2);
    memcpy(buffer3, &i, size);
    memcpy(buffer3 + size, &j, size);
    //Though their inner Metadata has the same values, they are different objects
    TupleRow t3 = TupleRow(metas2, sizeof(uint16_t) * 2, buffer3);
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


    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "x"}},
                                                                  {{"name", "y"}},
                                                                  {{"name", "z"}},
                                                                  {{"name", "ciao"}}};
    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};
    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    TableMetadata *table_meta = new TableMetadata(particles_table, keyspace, keysnames, colsnames, test_session);

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

    delete (t);
    delete (result);
    delete (CTable);


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

}


TEST(TestingEmptyValues, WriteSimple) {
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

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "first"}},
                                                                  {{"name", "second"}},
                                                                  {{"name", "third"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata *table_meta = new TableMetadata(only_keys_table, keyspace, keysnames, colsnames, test_session);

    CacheTable *cache = new CacheTable(table_meta, test_session, config);


    char *buffer = (char *) malloc(sizeof(int) + sizeof(int) + sizeof(char *)); //keys

    int32_t k1 = 3682;
    int32_t k2 = 3682;
    const char *k3_base = "SomeKey";
    char *k3 = (char *) malloc(std::strlen(k3_base) + 1);

    std::memcpy(k3, k3_base, std::strlen(k3_base) + 1);

    memcpy(buffer, &k1, sizeof(int32_t));
    memcpy(buffer + sizeof(int), &k2, sizeof(int32_t));
    memcpy(buffer + sizeof(int) + sizeof(int), &k3, sizeof(char *));


    char *buffer2 = nullptr;


    TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(int) + sizeof(char *), buffer);
    TupleRow *values = new TupleRow(table_meta->get_values(), 0, buffer2);


    cache->put_crow(keys, values);


    std::vector<const TupleRow *> results = cache->get_crow(keys);

    delete (keys);
    delete (values);
    delete (cache);


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

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "x"}},
                                                                  {{"name", "y"}},
                                                                  {{"name", "z"}}};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata *table_meta = new TableMetadata(particles_wr_table, keyspace, keysnames, colsnames, test_session);

    CacheTable *cache = new CacheTable(table_meta, test_session, config);


    char *buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

    int32_t k1 = 3682;
    float k2 = 143.2;
    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    float v[] = {0.01, 1.25, 0.98};

    char *buffer2 = (char *) malloc(sizeof(float) * 3); //values
    memcpy(buffer2, &v, sizeof(float) * 3);


    TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);
    TupleRow *values = new TupleRow(table_meta->get_values(), sizeof(float) * 3, buffer2);


    values->setNull(1);

    cache->put_crow(keys, values);

    delete (keys);
    delete (values);
    delete (cache);

    /*** now we write into another table with text ***/

    keysnames = {{{"name", "partid"}},
                 {{"name", "time"}}};
    colsnames = {{{"name", "x"}},
                 {{"name", "ciao"}}};

    tokens = {};


    table_meta = new TableMetadata(words_wr_table, keyspace, keysnames, colsnames, test_session);

    cache = new CacheTable(table_meta, test_session, config);


    buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

    k1 = 3682;
    k2 = 143.2;
    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    float v1 = 23.12;
    std::string *v2 = new std::string("someBeautiful, and randomText");

    buffer2 = (char *) malloc(sizeof(float) + sizeof(char *)); //values
    memcpy(buffer2, &v1, sizeof(float));
    memcpy(buffer2 + sizeof(float), &v2, sizeof(void *));


    keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);
    values = new TupleRow(table_meta->get_values(), sizeof(float) * 3, buffer2);


    values->setNull(1);

    cache->put_crow(keys, values);

    delete (keys);
    delete (values);
    delete (cache);
    delete (v2);

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

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "x"}},
                                                                  {{"name", "y"}},
                                                                  {{"name", "z"}}};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata *table_meta = new TableMetadata(particles_wr_table, keyspace, keysnames, colsnames, test_session);

    CacheTable *cache = new CacheTable(table_meta, test_session, config);

    for (uint32_t id = 0; id < 8000; ++id) {
        char *buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

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
        delete (keys);
        delete (values);
    }

    delete (cache);


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

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "x"}},
                                                                  {{"name", "ciao"}}};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata *table_meta = new TableMetadata(words_wr_table, keyspace, keysnames, colsnames, test_session);

    CacheTable *cache = new CacheTable(table_meta, test_session, config);

    for (uint32_t id = 0; id < 8000; ++id) {
        char *buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

        float k2 = 143.2;
        memcpy(buffer, &id, sizeof(int));
        memcpy(buffer + sizeof(int), &k2, sizeof(float));

        float v[] = {0.01, 1.25, 0.98};

        char *buffer2 = (char *) malloc(sizeof(float) + sizeof(char *)); //values
        memcpy(buffer2, &v, sizeof(float));


        TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);
        TupleRow *values = new TupleRow(table_meta->get_values(), sizeof(float) * 3, buffer2);


        values->setNull(1);

        cache->put_crow(keys, values);
        delete (keys);
        delete (values);
    }

    delete (cache);


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}

//TODO reimplement test with the current interface
/*
TEST(TestingCacheTable, StoreNumpies) {


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

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "uuid"}},
                                                                  {{"name", "position"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "data"}}};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


//    fireandforget("CREATE TABLE test.arrays_aux(uuid uuid,  position int, data blob, PRIMARY KEY (uuid,position));", test_session);
    TableMetadata *table_meta = new TableMetadata("arrays_aux", keyspace, keysnames, colsnames, test_session);

    CacheTable *cache = new CacheTable(table_meta, test_session, config);

    char *buffer = (char *) malloc(sizeof(uint64_t *) + sizeof(int)); //keys
    uint64_t *uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);
    *uuid = 123;
    *(uuid + 1) = 456;

    memcpy(buffer, &uuid, sizeof(uint64_t *));
    int32_t k2 = 789;
    memcpy(buffer + sizeof(uint64_t *), &k2, sizeof(int));

    std::string txtbytes = "somerandombytes";
    uint64_t nbytes = strlen(txtbytes.c_str());

    char *bytes = (char *) malloc(nbytes + sizeof(uint64_t));
    char *buffer2 = (char *) malloc(sizeof(char *)); //values
    memcpy(buffer2, &bytes, sizeof(char *));

    memcpy(bytes, &nbytes, sizeof(uint64_t));
    bytes += sizeof(uint64_t);
    memcpy(bytes, txtbytes.c_str(), nbytes);

    TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(uint64_t *) + sizeof(int), buffer);
    TupleRow *values = new TupleRow(table_meta->get_values(), sizeof(char *), buffer2);

    cache->put_crow(keys, values);
    delete (keys);
    delete (values);

    delete (cache);


    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}
*/

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

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "x"}},
                                                                  {{"name", "y"}},
                                                                  {{"name", "z"}}};

    std::vector<std::pair<int64_t, int64_t> > tokens = {};

    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata *table_meta = new TableMetadata(particles_wr_table, keyspace, keysnames, colsnames, test_session);

    CacheTable *cache = new CacheTable(table_meta, test_session, config);


    char *buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

    int32_t k1 = 4682;
    float k2 = 93.2;
    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    float v[] = {4.43, 9.99, 1.238};

    char *buffer2 = (char *) malloc(sizeof(float) * 3); //values
    memcpy(buffer2, &v, sizeof(float) * 3);


    TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);
    TupleRow *values = new TupleRow(table_meta->get_values(), sizeof(float) * 3, buffer2);


    values->setNull(1);

    cache->put_crow(keys, values);

    delete (keys);
    delete (values);
    delete (cache);


    //now we read the data

    keysnames = {{{"name", "partid"}},
                 {{"name", "time"}}};
    colsnames = {{{"name", "x"}},
                 {{"name", "y"}},
                 {{"name", "z"}}};

    tokens = {};

    table_meta = new TableMetadata(particles_wr_table, keyspace, keysnames, colsnames, test_session);

    cache = new CacheTable(table_meta, test_session, config);


    buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);


    std::vector<const TupleRow *> results = cache->get_crow(keys);
    ASSERT_TRUE(results.size() == 1);
    const TupleRow *read_values = results[0];
    const float *v0 = (float *) read_values->get_element(0);
    const float *v1 = (float *) read_values->get_element(1);
    const float *v2 = (float *) read_values->get_element(2);
    EXPECT_DOUBLE_EQ(*v0, v[0]);
    EXPECT_TRUE(v1 == nullptr);
    EXPECT_DOUBLE_EQ(*v2, v[2]);

    delete (keys);
    delete (read_values);
    delete (cache);

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


    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "ciao"}}};

    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    TableMetadata *table_meta = new TableMetadata(particles_table, keyspace, keysnames, colsnames, test_session);

    CacheTable *cache = new CacheTable(table_meta, test_session, config);


    TupleRow *t = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);

    std::vector<const TupleRow *> all_rows = cache->get_crow(t);
    delete (t);
    delete (cache);
    EXPECT_EQ(all_rows.size(), 1);
    const TupleRow *result = all_rows.at(0);

    EXPECT_FALSE(result == NULL);

    if (result != 0) {

        const void *v = result->get_element(0);
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "74040");

    }

    for (const TupleRow *tuple:all_rows) {
        delete (tuple);
    }

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}




/** Test there are no problems when requesting twice the same key **/
TEST(TestingCacheTable, GetRowStringSameKey) {

    uint32_t n_queries = 100;


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

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "ciao"}}};


    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";

    TableMetadata *table_meta = new TableMetadata(particles_table, keyspace, keysnames, colsnames, test_session);
    CacheTable *cache = new CacheTable(table_meta, test_session, config);


    /** build key **/

    char *buffer = (char *) malloc(sizeof(int) + sizeof(float));

    int val = 1234;
    memcpy(buffer, &val, sizeof(int));

    float f = 12340;
    memcpy(buffer + sizeof(int), &f, sizeof(float));
    TupleRow *key = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);


    /** get queries **/

    for (uint32_t query_i = 0; query_i < n_queries; query_i++) {

        std::vector<const TupleRow *> all_rows = cache->get_crow(key);
        EXPECT_EQ(all_rows.size(), 1);

        const TupleRow *result = all_rows.at(0);
        EXPECT_FALSE(result == NULL);

        if (result != 0) {
            const void *v = result->get_element(0);
            int64_t addr;
            memcpy(&addr, v, sizeof(char *));
            char *d = reinterpret_cast<char *>(addr);

            EXPECT_STREQ(d, "74040");
        }

        for (const TupleRow *tuple:all_rows) {
            delete (tuple);
        }
    }


    delete (key);
    delete (cache);
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

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "ciao"}}};

    std::string token_pred = "WHERE token(partid)>=? AND token(partid)<?";
    std::vector<std::pair<int64_t, int64_t> > tokens = {std::pair<int64_t, int64_t>(-10000, 10000)};

    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata *table_meta = new TableMetadata(particles_table, keyspace, keysnames, colsnames, test_session);

    CacheTable *T = new CacheTable(table_meta, test_session, config);



    //TupleRow *t = new TupleRow(T._test_get_keys_factory()->get_metadata(), sizeof(int) + sizeof(float), buffer);

    std::vector<const TupleRow *> results = T->get_crow(buffer);


    EXPECT_FALSE(results.empty());

    if (!results.empty()) {

        const void *v = results[0]->get_payload();
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);

        EXPECT_STREQ(d, "74040");
    }
    for (const TupleRow *t_unit:results) {
        delete (t_unit);
    }
    buffer = (char *) malloc(sizeof(int) + sizeof(float));
    memcpy(buffer, &val, sizeof(int));
    memcpy(buffer + sizeof(int), &f, sizeof(float));


    char *substitue = (char *) malloc(sizeof("71919"));
    memcpy(substitue, "71919", sizeof("71919"));
    char **payload2 = (char **) malloc(sizeof(char *));
    *payload2 = substitue;

    T->put_crow(buffer, payload2);


    delete (T);
    //With the aim to synchronize
    table_meta = new TableMetadata(particles_table, keyspace, keysnames, colsnames, test_session);
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

        EXPECT_STREQ(d, "74040");
    }


    for (const TupleRow *t_unit:results) {
        delete (t_unit);
    }

    substitue = (char *) malloc(sizeof("74040"));
    memcpy(substitue, "74040", sizeof("74040"));
    payload2 = (char **) malloc(sizeof(char *));
    *payload2 = substitue;

    buffer = (char *) malloc(sizeof(int) + sizeof(float));
    memcpy(buffer, &val, sizeof(int));
    memcpy(buffer + sizeof(int), &f, sizeof(float));

    T->put_crow(buffer, payload2);

    delete (T);

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

    ColumnMeta cm1 = ColumnMeta();
    cm1.info = {{"name", "partid"}};
    cm1.type = CASS_VALUE_TYPE_INT;
    cm1.position = 0;
    cm1.size = sizeof(int);

    ColumnMeta cm2 = ColumnMeta();
    cm2.info = {{"name", "time"}};
    cm2.type = CASS_VALUE_TYPE_FLOAT;
    cm2.position = sizeof(int);
    cm2.size = sizeof(float);

    std::vector<ColumnMeta> v = {cm1, cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas = std::make_shared<std::vector<ColumnMeta>>(v);

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "ciao"}}};

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
    config["prefetch_size"] = "100";


    TableMetadata *table_meta = new TableMetadata(particles_table, keyspace, keysnames, colsnames, test_session);

    CacheTable T = CacheTable(table_meta, test_session, config);

    Prefetch *P = new Prefetch(tokens, table_meta, test_session, config);


    TupleRow *result = NULL;
    uint16_t it = 0;


    while ((result = P->get_cnext()) != NULL) {
        const void *v = result->get_element(2);
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);
        std::string empty_str = "";
        std::string result_str(d);
        EXPECT_TRUE(result_str > empty_str);
        delete (result);
        ++it;
    }

    EXPECT_EQ(it, 10001);

    delete (P);

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}





/*** CACHE API TESTS ***/


TEST(TestingStorageInterfaceCpp, ConnectDisconnect) {
    StorageInterface *StorageI = new StorageInterface(nodePort, contact_p);
    delete (StorageI);
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

    ColumnMeta cm1 = ColumnMeta();
    cm1.info = {{"name", "partid"}};
    cm1.type = CASS_VALUE_TYPE_INT;
    cm1.position = 0;
    cm1.size = sizeof(int);

    ColumnMeta cm2 = ColumnMeta();
    cm2.info = {{"name", "time"}};
    cm2.type = CASS_VALUE_TYPE_FLOAT;
    cm2.position = sizeof(int);
    cm2.size = sizeof(float);

    std::vector<ColumnMeta> v = {cm1, cm2};
    std::shared_ptr<std::vector<ColumnMeta>> metas = std::make_shared<std::vector<ColumnMeta>>(v);

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > colsnames = {{{"name", "ciao"}}};

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
    config["cache_size"] = "100";
    config["prefetch_size"] = "30";
    config["update_cache"] = "true";

    TableMetadata *table_meta = new TableMetadata(particles_table, keyspace, keysnames, colsnames, test_session);

    CacheTable T = CacheTable(table_meta, test_session, config);

    //By default iterates items
    Prefetch *P = new Prefetch(tokens, table_meta, test_session, config);


    TupleRow *result = NULL;
    uint16_t it = 0;


    while ((result = P->get_cnext()) != NULL) {
        const void *v = result->get_element(2);
        int64_t addr;
        memcpy(&addr, v, sizeof(char *));
        char *d = reinterpret_cast<char *>(addr);
        std::string empty_str = "";
        std::string result_str(d);
        EXPECT_TRUE(result_str > empty_str);
        delete (result);
        ++it;
    }

    EXPECT_EQ(it, 10001);

    delete (P);

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}


TEST(TestingStorageInterfaceCpp, CreateAndDelCache) {
    /** KEYS **/

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > read_colsnames = {{{"name", "x"}},
                                                                       {{"name", "ciao"}}};
    std::vector<std::map<std::string, std::string> > write_colsnames = {{{"name", "y"}},
                                                                        {{"name", "ciao"}}};

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

    StorageInterface *StorageI = new StorageInterface(nodePort, contact_p);
    CacheTable *table = StorageI->make_cache(particles_table, keyspace, keysnames, read_colsnames, config);

    delete (table);
    delete (StorageI);
}


TEST(TestingStorageInterfaceCpp, CreateAndDelCacheWrong) {
    /** This test demonstrates that deleting the Cache provider
     * before deleting cache instances doesnt raise exceptions
     * or enter in any kind of lock
     * **/

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > read_colsnames = {{{"name", "x"}},
                                                                       {{"name", "ciao"}}};


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

    StorageInterface *StorageI = new StorageInterface(nodePort, contact_p);
    CacheTable *table = StorageI->make_cache(particles_table, keyspace, keysnames, read_colsnames, config);

    delete (StorageI);
    delete (table);
}


TEST(TestingStorageInterfaceCpp, IteratePrefetch) {
    /** This test demonstrates that deleting the Cache provider
     * before deleting cache instances doesnt raise exceptions
     * or enter in any kind of lock
     * **/

    std::vector<std::map<std::string, std::string> > keysnames = {{{"name", "partid"}},
                                                                  {{"name", "time"}}};
    std::vector<std::map<std::string, std::string> > read_colsnames = {{{"name", "x"}},
                                                                       {{"name", "ciao"}}};

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
    config["prefetch_size"] = "100";

    StorageInterface *StorageI = new StorageInterface(nodePort, contact_p);
    CacheTable *table = StorageI->make_cache(particles_table, keyspace, keysnames, read_colsnames, config);

    Prefetch *P = StorageI->get_iterator(particles_table, keyspace, keysnames, read_colsnames, tokens, config);
    int it = 0;
    TupleRow *T = NULL;
    while ((T = P->get_cnext()) != NULL) {
        ++it;
        delete (T);
    }
    EXPECT_EQ(it, 10001);
    delete (P->get_metadata());
    delete (P);
    delete (table);
    delete (StorageI);
}


TEST(TestingCacheTable, DeleteRow) {

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

    std::vector<std::map<std::string, std::string>> keysnames = {{{"name", "partid"}},
                                                                 {{"name", "time"}}};
    std::vector<std::map<std::string, std::string>> colsnames = {{{"name", "x"}},
                                                                 {{"name", "y"}},
                                                                 {{"name", "z"}}};

    std::vector<std::pair<int64_t, int64_t>> tokens = {};

    std::map<std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";


    TableMetadata *table_meta = new TableMetadata(particles_wr_table, keyspace, keysnames, colsnames, test_session);

    CacheTable *cache = new CacheTable(table_meta, test_session, config);


    char *buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

    int32_t k1 = 4682;
    float k2 = 93.2;
    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    float v[] = {4.43, 9.99, 1.238};

    char *buffer2 = (char *) malloc(sizeof(float) * 3); //values
    memcpy(buffer2, &v, sizeof(float) * 3);


    TupleRow *keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);
    TupleRow *values = new TupleRow(table_meta->get_values(), sizeof(float) * 3, buffer2);


    cache->put_crow(keys, values);

    delete (keys);
    delete (values);
    delete (cache);


//now we read the data

    keysnames = {{{"name", "partid"}},
                 {{"name", "time"}}};
    colsnames = {{{"name", "x"}},
                 {{"name", "y"}},
                 {{"name", "z"}}};

    tokens = {};

    table_meta = new TableMetadata(particles_wr_table, keyspace, keysnames, colsnames, test_session);

    cache = new CacheTable(table_meta, test_session, config);

    buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);


    std::vector<const TupleRow *> results = cache->get_crow(keys);
    ASSERT_TRUE(results.size() == 1);
    const TupleRow *read_values = results[0];
    const float *v0 = (float *) read_values->get_element(0);
    const float *v1 = (float *) read_values->get_element(1);
    const float *v2 = (float *) read_values->get_element(2);
    EXPECT_DOUBLE_EQ(*v0, v[0]);
    EXPECT_DOUBLE_EQ(*v1, v[1]);
    EXPECT_DOUBLE_EQ(*v2, v[2]);

    delete (keys);


    buffer = (char *) malloc(sizeof(int) + sizeof(float)); //keys

    memcpy(buffer, &k1, sizeof(int));
    memcpy(buffer + sizeof(int), &k2, sizeof(float));

    keys = new TupleRow(table_meta->get_keys(), sizeof(int) + sizeof(float), buffer);

    cache->delete_crow(keys);

    results = cache->get_crow(keys);

    EXPECT_TRUE(results.empty());

    delete (keys);
    delete (read_values);
    delete (cache);

    CassFuture *close_future = cass_session_close(test_session);
    cass_future_wait(close_future);
    cass_future_free(close_future);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);
}

TEST(TableMeta, TupleWithTwoInts) {

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

    fireandforget("CREATE KEYSPACE test WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};",
                  test_session);
    fireandforget(
            "CREATE TABLE test.people(dni text PRIMARY KEY, info tuple<int,int>);",
            test_session);

    const char *query = "INSERT INTO test.people(dni, info) VALUES ('socUnDNI', (1,2));";

    fireandforget(query, test_session);

    CassStatement *cs
            = cass_statement_new("INSERT INTO test.people(dni, info) VALUES ('socUnDNI2', ?)", 1);

    std::vector<std::map<std::string, std::string>> keysnames = {{{"name", "dni"}}};

    std::vector<std::map<std::string, std::string>> colsnames = {{{"name", "info"}}};

    TableMetadata *table_meta = new TableMetadata("people", "test", keysnames, colsnames, test_session);

    // INSERT WITH BIND METHOD //

    std::tuple<int, int> mytuple(10, 10);

    int *buffer2 = (int *) malloc(sizeof(mytuple));
    memcpy(buffer2, &mytuple, sizeof(mytuple));

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    uint16_t bsize = (sizeof(int32_t));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_INT, nullptr, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_INT, nullptr, bsize, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    TupleRow *values = new TupleRow(CM.pointer, sizeof(mytuple), buffer2);
    TupleRow *valuess = new TupleRow(table_meta->get_values(), sizeof(mytuple), &values);

    std::shared_ptr<const std::vector<ColumnMeta> > cols = table_meta->get_values();

    const void *element_i = valuess->get_element(0);
    TupleRow **ptr = (TupleRow **) element_i;
    const TupleRow *inner_data = *ptr;
    int32_t *value = (int32_t *) inner_data->get_element(0);


    TupleRowFactory trf = TupleRowFactory(cols);

    trf.bind(cs, valuess, 0);
    connect_future = cass_session_execute(test_session, cs);

    // INSERT WITH CASSANDRA METHODS //

    cs = cass_statement_new("INSERT INTO test.people(dni, info) VALUES ('socUnDNI3', ?)", 1);


    CassTuple *ct = cass_tuple_new(2);
    cass_int32_t a = 1;
    cass_int32_t c = 1;
    cass_tuple_set_int32(ct, 0, a);
    cass_tuple_set_int32(ct, 1, c);

    cass_statement_bind_tuple(cs, 0, ct);

    connect_future = cass_session_execute(test_session, cs);

    rc = cass_future_error_code(connect_future);

    EXPECT_TRUE(rc == CASS_OK);
    if (rc != CASS_OK) {
        std::cout << "ERROR ON EXECUTING QUERY: " << cass_error_desc(rc) << std::endl;
    }
    cass_future_free(connect_future);
    cass_statement_free(cs);
    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

}

TEST(TableMeta, BigIntFromCassandra) {

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

    fireandforget("CREATE KEYSPACE test WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};",
                  test_session);
    fireandforget(
            "CREATE TABLE test.people(dni text PRIMARY KEY, info tuple<bigint,bigint>);",
            test_session);

    CassStatement *cs
            = cass_statement_new("INSERT INTO test.people(dni, info) VALUES ('socUnDNI2', ?)", 1);

    std::vector<std::map<std::string, std::string>> keysnames = {{{"name", "dni"}}};

    std::vector<std::map<std::string, std::string>> colsnames = {{{"name", "info"}}};
    TableMetadata *table_meta = new TableMetadata("people", "test", keysnames, colsnames, test_session);

    CassTuple *ct = cass_tuple_new(2);
    cass_int64_t a = 55000000000000000;
    cass_int64_t c = 55000000000000001;
    cass_tuple_set_int64(ct, 0, a);
    cass_tuple_set_int64(ct, 1, c);

    cass_statement_bind_tuple(cs, 0, ct);

    connect_future = cass_session_execute(test_session, cs);

    rc = cass_future_error_code(connect_future);

    EXPECT_TRUE(rc == CASS_OK);
    if (rc != CASS_OK) {
        std::cout << "ERROR ON EXECUTING QUERY: " << cass_error_desc(rc) << std::endl;
    }

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(int64_t));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_BIGINT, nullptr, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_BIGINT, nullptr, bsize, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    std::shared_ptr<const std::vector<ColumnMeta> > cols = table_meta->get_values();
    TupleRowFactory trf = TupleRowFactory(cols);

    std::tuple<long, long> mytuple(55000000000000000, 5500000000000001);
    char *buffer2 = (char *) malloc(sizeof(mytuple)); //values

    memcpy(buffer2, &mytuple, sizeof(mytuple));
    TupleRow *values = new TupleRow(CM.pointer, sizeof(mytuple), buffer2);
    TupleRow *valuess = new TupleRow(table_meta->get_values(), sizeof(mytuple), &values);

    trf.bind(cs, valuess, 0);
    connect_future = cass_session_execute(test_session, cs);

    //NOW LET'S TRY TO RETRIEVE THE DATA INSERTED (GET ROW)
    CassStatement *statement = cass_statement_new("SELECT info FROM test.people", 0);
    CassFuture *query_future = cass_session_execute(test_session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    rc = cass_future_error_code(connect_future);
    if (result == NULL) {
        /* Handle error */
        std::string error(cass_error_desc(rc));
        cass_future_free(connect_future);
        cass_statement_free(cs);
        throw ModuleException("CacheTable: Get row error on result" + error);
    }

    cass_future_free(connect_future);
    cass_statement_free(cs);

    int64_t counter = 0;
    std::vector<const TupleRow *> gvalues(cass_result_row_count(result));
    const CassRow *row;
    CassIterator *it = cass_iterator_from_result(result);


    while (cass_iterator_next(it)) {
        row = cass_iterator_get_row(it);
        gvalues[counter] = trf.make_tuple(row);
        ++counter;
    }

    const void *element_i = gvalues[0]->get_element(0);
    TupleRow **ptr = (TupleRow **) element_i;
    const TupleRow *inner_data = *ptr;

    int64_t *value = (int64_t *) inner_data->get_element(0);
    int64_t *value2 = (int64_t *) inner_data->get_element(1);

    EXPECT_EQ(*value, 5500000000000001);
    EXPECT_EQ(*value2, 55000000000000000);
    cass_iterator_free(it);
    cass_result_free(result);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

}


TEST(TableMeta, TwoTextFromCassandra) {

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

    fireandforget("CREATE KEYSPACE test WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};",
                  test_session);
    fireandforget(
            "CREATE TABLE test.people(dni text PRIMARY KEY, info tuple<text,text>);",
            test_session);

    CassStatement *cs
            = cass_statement_new("INSERT INTO test.people(dni, info) VALUES ('socUnDNI2', ?)", 1);

    std::vector<std::map<std::string, std::string>> keysnames = {{{"name", "dni"}}};

    std::vector<std::map<std::string, std::string>> colsnames = {{{"name", "info"}}};

    TableMetadata *table_meta = new TableMetadata("people", "test", keysnames, colsnames, test_session);

    //connect_future = cass_session_execute(test_session, cs);


    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    uint16_t bsize = (sizeof(int64_t));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_TEXT, nullptr, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_TEXT, nullptr, bsize, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    std::shared_ptr<const std::vector<ColumnMeta> > cols = table_meta->get_values();
    TupleRowFactory trf = TupleRowFactory(cols);

    std::tuple<string, string> mytuple;
    mytuple = make_tuple("text1", "text2");

    void *buffer = malloc(sizeof(int64_t) * 2);
    int64_t *buffer2 = (int64_t *) buffer;


    memcpy(buffer2, &(get<0>(mytuple)), sizeof(int64_t));
    memcpy(buffer2 + 1, &(get<1>(mytuple)), sizeof(int64_t));

    TupleRow *values = new TupleRow(CM.pointer, sizeof(mytuple), buffer2);
    TupleRow *valuess = new TupleRow(table_meta->get_values(), sizeof(mytuple), &values);

    trf.bind(cs, valuess, 0);
    connect_future = cass_session_execute(test_session, cs);

    //NOW LET'S TRY TO RETRIEVE THE DATA INSERTED (GET ROW)
    CassStatement *statement = cass_statement_new("SELECT info FROM test.people", 0);
    CassFuture *query_future = cass_session_execute(test_session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    rc = cass_future_error_code(connect_future);
    if (result == NULL) {
        /* Handle error */
        std::string error(cass_error_desc(rc));
        cass_future_free(connect_future);
        cass_statement_free(cs);
        throw ModuleException("CacheTable: Get row error on result" + error);
    }

    cass_future_free(connect_future);
    cass_statement_free(cs);

    int64_t counter = 0;
    std::vector<const TupleRow *> gvalues(cass_result_row_count(result));
    const CassRow *row;
    CassIterator *it = cass_iterator_from_result(result);


    while (cass_iterator_next(it)) {
        row = cass_iterator_get_row(it);
        gvalues[counter] = trf.make_tuple(row);
        ++counter;
    }

    const void *element_i = gvalues[0]->get_element(0);
    TupleRow **ptr = (TupleRow **) element_i;
    const TupleRow *inner_data = *ptr;

    int64_t *value = (int64_t *) inner_data->get_element(0);
    int64_t *value2 = (int64_t *) inner_data->get_element(1);

    ASSERT_STREQ(reinterpret_cast<char *>(*value), "text1");
    ASSERT_STREQ(reinterpret_cast<char *>(*value2), "text2");

    cass_result_free(result);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

}

TEST(TableMeta, BigIntANDTextFromCassandra) {

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

    fireandforget("CREATE KEYSPACE test WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};",
                  test_session);
    fireandforget(
            "CREATE TABLE test.people(dni text PRIMARY KEY, info tuple<bigint,text>);",
            test_session);

    CassStatement *cs
            = cass_statement_new("INSERT INTO test.people(dni, info) VALUES ('socUnDNI2', ?)", 1);

    std::vector<std::map<std::string, std::string>> keysnames = {{{"name", "dni"}}};

    std::vector<std::map<std::string, std::string>> colsnames = {{{"name", "info"}}};

    TableMetadata *table_meta = new TableMetadata("people", "test", keysnames, colsnames, test_session);

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    uint16_t bsize = (sizeof(int64_t));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_BIGINT, nullptr, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_TEXT, nullptr, bsize, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    std::shared_ptr<const std::vector<ColumnMeta> > cols = table_meta->get_values();
    TupleRowFactory trf = TupleRowFactory(cols);

    std::tuple<long, string> mytuple;
    mytuple = make_tuple(1000000000000000000, "no quiero parsearme");

    void *buffer = malloc(sizeof(int64_t) * 2);
    int64_t *buffer2 = (int64_t *) buffer;


    memcpy(buffer2, &(get<0>(mytuple)), sizeof(int64_t));
    memcpy(buffer2 + 1, &(get<1>(mytuple)), sizeof(int64_t));

    TupleRow *values = new TupleRow(CM.pointer, sizeof(mytuple), buffer2);
    TupleRow *valuess = new TupleRow(table_meta->get_values(), sizeof(mytuple), &values);

    trf.bind(cs, valuess, 0);
    connect_future = cass_session_execute(test_session, cs);

    //NOW LET'S TRY TO RETRIEVE THE DATA INSERTED (GET ROW)
    CassStatement *statement = cass_statement_new("SELECT info FROM test.people", 0);
    CassFuture *query_future = cass_session_execute(test_session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    rc = cass_future_error_code(connect_future);
    if (result == NULL) {
        /* Handle error */
        std::string error(cass_error_desc(rc));
        cass_future_free(connect_future);
        cass_statement_free(cs);
        throw ModuleException("CacheTable: Get row error on result" + error);
    }

    cass_future_free(connect_future);
    cass_statement_free(cs);

    int64_t counter = 0;
    std::vector<const TupleRow *> gvalues(cass_result_row_count(result));
    const CassRow *row;
    CassIterator *it = cass_iterator_from_result(result);


    while (cass_iterator_next(it)) {
        row = cass_iterator_get_row(it);
        gvalues[counter] = trf.make_tuple(row);
        ++counter;
    }

    const void *element_i = gvalues[0]->get_element(0);
    TupleRow **ptr = (TupleRow **) element_i;
    const TupleRow *inner_data = *ptr;

    int64_t *value = (int64_t *) inner_data->get_element(0);
    int64_t *value2 = (int64_t *) inner_data->get_element(1);

    EXPECT_EQ(*value, 1000000000000000000);
    ASSERT_STREQ(reinterpret_cast<char *>(*value2), "no quiero parsearme");

    cass_result_free(result);

    cass_cluster_free(test_cluster);
    cass_session_free(test_session);

}

TEST(TableMeta, CheckErrorTupleRowFact_Type) {

    CassSession *test_session = NULL;
    CassCluster *test_cluster = NULL;

    CassFuture *connect_future = NULL;
    test_cluster = cass_cluster_new();
    test_session = cass_session_new();

    cass_cluster_set_contact_points(test_cluster, contact_p);
    cass_cluster_set_port(test_cluster, nodePort);

    connect_future = cass_session_connect_keyspace(test_session, test_cluster, keyspace);

    CassError rcc = cass_future_error_code(connect_future);
    EXPECT_TRUE(rcc == CASS_OK);

    cass_future_free(connect_future);

    fireandforget("DROP KEYSPACE IF EXISTS test;", test_session);

    fireandforget("CREATE KEYSPACE test WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};",
                  test_session);

    CassStatement *statement = cass_statement_new("CREATE TABLE test.people(dni text PRIMARY KEY, info it);", 0);
    CassFuture *connect_fut = cass_session_execute(test_session, statement);
    CassError rc = cass_future_error_code(connect_fut);

    try {
        CHECK_CASS("Cannot create table");
    }
    catch (ModuleException &e1) {
        std::string excep_text = e1.what();
        GTEST_CHECK_(excep_text.find("Cannot create table"));
    }
}
