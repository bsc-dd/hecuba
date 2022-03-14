#ifndef HFETCH_WRITER_H
#define HFETCH_WRITER_H

#define MAX_ERRORS 10

#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <functional>

#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_hash_map.h"

#include "TimestampGenerator.h"
#include "TupleRowFactory.h"


class Writer {
public:
    Writer(const TableMetadata *table_meta, CassSession *session,
           std::map<std::string, std::string> &config);

    ~Writer();

    void set_timestamp_gen(TimestampGenerator *time_gen);

    bool call_async();

    void flush_elements();

    void write_to_cassandra(const TupleRow *keys, const TupleRow *values);

    void write_to_cassandra(void *keys, void *values);

    // Overload 'write_to_casandra' to write a single column (instead of all the columns)
    void write_to_cassandra(void *keys, void *values , const char *value_name);

    void wait_writes_completion(void);

    void set_error_occurred(std::string error, const void *keys, const void *values);

    const TableMetadata *get_metadata() {
        return table_metadata;
    }
    void enable_lazy_write(void);
    void disable_lazy_write(void);

private:
    struct HashCompare {    // This is used for lazy_write, currently only
                            // StorageNumpys are supported. The hash generator
                            // must be generalized to support other types.
        static size_t hash( const TupleRow* key ) {
            void * data= key->get_payload();
            size_t key_length=2*sizeof(uint64_t)+2*sizeof(uint32_t);
            void *key_content=malloc(key_length);
            memcpy (key_content,(void *) (*(char **)data), 2*sizeof(uint64_t)); //* copy the storage id
            memcpy (((char*)key_content)+2*sizeof(uint64_t), (void *)(((char*)data)+sizeof(uint64_t)), 2*sizeof(uint32_t)); //* copy the cluster_id and the block_id
            auto tmp= std::hash<std::string>{}(std::string((char *) key_content, key_length));
            free (key_content);
            return tmp;
        }
        static bool equal(const TupleRow* key1, const TupleRow* key2) {
            bool tmp = (hash(key1) == hash(key2));
            return tmp;
        }
    };

    CassSession *session;

/** ownership **/

    const CassPrepared *prepared_query;

    TupleRowFactory *k_factory;
    TupleRowFactory *v_factory;

    bool lazy_write_enabled;
    tbb::concurrent_hash_map <const TupleRow *, const TupleRow *, HashCompare> *dirty_blocks;
    tbb::concurrent_bounded_queue <std::pair<const TupleRow *, const TupleRow *>> data;

    uint32_t max_calls;
    std::atomic<uint32_t> ncallbacks;
    std::atomic<uint32_t> error_count;
    const TableMetadata *table_metadata;

    bool disable_timestamps;
    TimestampGenerator *timestamp_gen;


    void flush_dirty_blocks();
    void async_query_execute(const TupleRow *keys, const TupleRow *values);
    void queue_async_query( const TupleRow *keys, const TupleRow *values);
    static void callback(CassFuture *future, void *ptr);
    std::mutex async_query_thread_lock;
    bool async_query_thread_created;
    void async_query_thread_code();
    bool finish_async_query_thread;
    std::thread async_query_thread;
};


#endif //HFETCH_WRITER_H
