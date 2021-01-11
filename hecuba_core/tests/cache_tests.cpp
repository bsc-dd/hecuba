#include <iostream>
#include <cmath>
#include "gtest/gtest.h"
#include "../src/KVCache.h"
#include "../src/TupleRow.h"

/** TEST SETUP **/


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/***
 * Ensure the cache size stays below max size
 */
TEST(TestCache, VerifyMaxSize) {

    size_t cache_size = std::pow(10, 2);
    uint64_t n_inserts = std::pow(10, 5);
    ASSERT_LE(cache_size, n_inserts);

    KVCache<uint64_t, uint64_t> TestCache(cache_size);


    EXPECT_EQ(TestCache.size(), size_t(0));

    for (uint64_t i = 0; i < n_inserts; ++i) {
        TestCache.add(i, i);
    }

    EXPECT_LE(TestCache.size(), cache_size);
}


/***
 * Ensure updates are performed
 */
TEST(TestCache, VerifyUpdates) {

    size_t cache_size = std::pow(10, 2);
    uint64_t n_inserts = std::pow(10, 5);
    ASSERT_LE(cache_size, n_inserts);

    KVCache<uint64_t, uint64_t> TestCache(cache_size);
    // Warm-up
    for (uint64_t i = 0; i < n_inserts; ++i) {
        TestCache.add(i, i);
    }

    // Update base_val number of elements, but leave unchanged_elem to the original value
    uint64_t base_val = 100;
    uint64_t unchanged_elem = 10;
    for (uint64_t i = n_inserts - cache_size; i < n_inserts - unchanged_elem; ++i) {
        TestCache.add(i, base_val);
    }

    // Check updated values
    for (uint64_t i = n_inserts - cache_size; i < n_inserts - unchanged_elem; ++i) {
        EXPECT_EQ(TestCache.get(i), base_val);
    }

    // Check unmodified values
    for (uint64_t i = n_inserts - unchanged_elem; i < n_inserts; ++i) {
        EXPECT_EQ(TestCache.get(i), i);
    }
}


/***
 * Ensure removals are performed
 */
TEST(TestCache, VerifyRemovals) {

    size_t cache_size = std::pow(10, 2);
    uint64_t n_inserts = std::pow(10, 5);
    ASSERT_LE(cache_size, n_inserts);

    KVCache<uint64_t, uint64_t> TestCache(cache_size);

    // Warm-up
    for (uint64_t i = 0; i < n_inserts; ++i) {
        TestCache.add(i, i);
    }

    uint64_t n_removals = cache_size / 2;

    for (uint64_t i = n_inserts - cache_size; i < n_inserts - n_removals; ++i) {
        TestCache.remove(i);
    }

    // Should have been removed range(n_inserts-cache_size, n_inserts-n_removals)
    uint64_t ret;
    for (uint64_t i = n_inserts - cache_size; i < n_inserts - n_removals; ++i) {
        bool except_raised = false;
        try {
            ret = TestCache.get(i);
        }
        catch (std::out_of_range &ex) {
            except_raised = true;
        }
        EXPECT_TRUE(except_raised);
    }

    // Should be in cache range (n_inserts-n_removals, n_inserts)
    for (uint64_t i = n_inserts - n_removals; i < n_inserts; ++i) {
        bool except_raised = true;
        try {
            ret = TestCache.get(i);
        }
        catch (std::out_of_range &ex) {
            except_raised = false;
        }
        EXPECT_TRUE(except_raised);
    }
    EXPECT_EQ(ret, n_inserts - 1);
}



/***
 * Ensure all required data is kept
 */
TEST(TestCache, CheckAllDataIsPresent) {

    size_t cache_size = std::pow(10, 2);
    uint64_t n_inserts = cache_size;
    ASSERT_LE(cache_size, n_inserts);

    KVCache<uint64_t, uint64_t> TestCache(cache_size);

    for (uint64_t i = 0; i < n_inserts; ++i) {
        TestCache.add(i, i);
    }

    uint64_t ret;
    for (uint64_t i = 0; i < n_inserts; ++i) {
        ret = TestCache.get(i);
    }


    EXPECT_EQ(ret, n_inserts - 1);
    EXPECT_LE(TestCache.size(), cache_size);
}


/***
 * Ensure data replaced is not found with Tuples
 */
TEST(TestCache, VerifyLRUWorks) {
    uint64_t n_inserts = std::pow(10, 3);
    size_t cache_size = std::pow(10, 2);
    ASSERT_LE(cache_size, n_inserts);

    KVCache<TupleRow, TupleRow> TestCache(cache_size);

    std::map<std::string, std::string> info = {};
    std::vector<ColumnMeta> col_meta = {{info, CASS_VALUE_TYPE_BIGINT, nullptr, 0, sizeof(uint64_t)}};

    std::shared_ptr<const std::vector<ColumnMeta> > metas = std::make_shared<std::vector<ColumnMeta> >(col_meta);

    size_t payload_size = sizeof(uint64_t);


    for (uint64_t i = 0; i < n_inserts; ++i) {
        uint64_t *first = (uint64_t *) std::malloc(payload_size);
        *first = i;
        uint64_t *second = (uint64_t *) std::malloc(payload_size);
        *second = i;

        TupleRow key = TupleRow(metas, payload_size, first);
        TupleRow value = TupleRow(metas, payload_size, second);
        TestCache.add(key, value);
    }


    for (uint64_t i = n_inserts - cache_size; i < n_inserts; ++i) {
        uint64_t *first = (uint64_t *) std::malloc(payload_size);
        *first = i;
        TupleRow key = TupleRow(metas, payload_size, first);
        TupleRow ret = TestCache.get(key);

    }

    EXPECT_LE(TestCache.size(), cache_size);


    for (uint64_t i = 0; i < n_inserts - cache_size; ++i) {
        bool except_raised = false;

        uint64_t *first = (uint64_t *) std::malloc(payload_size);
        *first = i;
        TupleRow key = TupleRow(metas, payload_size, first);
        try {
            TupleRow ret = TestCache.get(key);
        }
        catch (std::out_of_range &ex) {
            except_raised = true;
        }
        EXPECT_TRUE(except_raised);
    }


    EXPECT_LE(TestCache.size(), cache_size);
}






/***
 * Ensure data replaced is deleted
 */
TEST(TestCache, VerifyDeleteIsCalled) {

    uint64_t n_inserts = 2;
    size_t cache_size = 1;
    ASSERT_LE(cache_size, n_inserts);

    KVCache<TupleRow, TupleRow> TestCache(cache_size);

    std::map<std::string, std::string> info = {};
    std::vector<ColumnMeta> col_meta = {{info, CASS_VALUE_TYPE_BIGINT, nullptr, 0, sizeof(uint64_t)}};

    std::shared_ptr<const std::vector<ColumnMeta> > metas = std::make_shared<std::vector<ColumnMeta> >(col_meta);

    size_t payload_size = sizeof(uint64_t);


    uint64_t i_val = 12345;


    uint64_t *first = (uint64_t *) std::malloc(payload_size);
    *first = i_val++;
    uint64_t *second = (uint64_t *) std::malloc(payload_size);
    *second = i_val++;

    TupleRow key = TupleRow(metas, payload_size, first);
    TupleRow value = TupleRow(metas, payload_size, second);
    TestCache.add(key, value);


    // Here we replaced the first key,value pair

    first = (uint64_t *) std::malloc(payload_size);
    *first = i_val++;
    second = (uint64_t *) std::malloc(payload_size);
    *second = i_val++;

    TupleRow key2 = TupleRow(metas, payload_size, first);
    TupleRow value2 = TupleRow(metas, payload_size, second);
    TestCache.add(key2, value2);


    EXPECT_EQ(key.use_count(), 1);
    EXPECT_EQ(value.use_count(), 1);
    EXPECT_GT(key2.use_count(), 1);
    EXPECT_GT(value2.use_count(), 1);
}





















/*** PERFORMANCE ***/

/***
 * Measure write performance
 */
TEST(TestCache, Write100k) {
    size_t cache_size = std::pow(10, 5);
    uint64_t n_inserts = std::pow(10, 5);

    KVCache<uint64_t, uint64_t> TestCache(cache_size);

    for (uint64_t i = 0; i < n_inserts; ++i) {
        TestCache.add(i, i);
    }

}


/***
 * Measure read performance
 */
TEST(TestCache, ReadAfterWrite100k) {
    size_t cache_size = std::pow(10, 5);
    uint64_t n_inserts = std::pow(10, 5);

    KVCache<uint64_t, uint64_t> TestCache(cache_size);

    for (uint64_t i = 0; i < n_inserts; ++i) {
        TestCache.add(i, i);
    }
    uint64_t ret;
    for (uint64_t i = 0; i < n_inserts; ++i) {
        ret = TestCache.get(i);
    }
    EXPECT_EQ(ret, n_inserts - 1);
}


/***
 * Measure mixed workload performance
 */
TEST(TestCache, MixedPerformance) {
    uint64_t n_inserts = std::pow(10, 5);
    uint64_t n_iter = 100;
    uint64_t iter_size = n_inserts / n_iter;
    double read_to_write_ratio = 0.2;
    uint64_t read_per_iter = iter_size * read_to_write_ratio;

    size_t cache_size = iter_size;
    KVCache<uint64_t, uint64_t> TestCache(cache_size);
    uint64_t ret = 0;
    for (uint64_t iter = 0; iter < n_iter; ++iter) {
        uint64_t start_i = iter * iter_size;
        for (uint64_t i = start_i; i < start_i + iter_size; ++i) {
            TestCache.add(i, i);
        }
        for (uint64_t i = start_i; i < start_i + read_per_iter; ++i) {
            ret = TestCache.get(i);
        }
    }

    EXPECT_GE(ret, size_t(0));
}

