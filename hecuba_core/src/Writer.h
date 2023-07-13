#ifndef HFETCH_WRITER_H
#define HFETCH_WRITER_H


#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <functional>
#include <librdkafka/rdkafka.h>

#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_hash_map.h"

#include "TimestampGenerator.h"
#include "TupleRowFactory.h"


class Writer {
public:
    Writer(const TableMetadata *table_meta, CassSession *session,
           std::map<std::string, std::string> &config);
    Writer(const Writer& src);
    Writer& operator =(const Writer& src);

    ~Writer();

    void set_timestamp_gen(TimestampGenerator *time_gen);


    void flush_elements();

    void write_to_cassandra(const TupleRow *keys, const TupleRow *values);

    void write_to_cassandra(void *keys, void *values);

    void enable_stream(const char* topic_name, std::map<std::string,std::string>  &config);

    rd_kafka_conf_t *create_stream_conf(std::map<std::string,std::string>  &config);


    void send_event(char* event, const uint64_t size);
    void send_event(const TupleRow* key,const TupleRow *value);
    void send_event(void* key, void* value);

    // Overload 'write_to_casandra' to write a single column (instead of all the columns)
    void write_to_cassandra(void *keys, void *values , const char *value_name);

    void wait_writes_completion(void);


    const TableMetadata *get_metadata() {
        return table_metadata;
    }
    void enable_lazy_write(void);
    void disable_lazy_write(void);

    CassStatement* bind_cassstatement(const TupleRow* keys, const TupleRow* values) const;
    void finish_async_call();
    CassSession* get_session() const;
    bool is_write_completed() const;

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

    CassSession *session = nullptr;

/** ownership **/

    const CassPrepared *prepared_query = nullptr;
    std::map<const std::string, const CassPrepared*> prepared_partial_queries;

    TupleRowFactory *k_factory = nullptr;
    TupleRowFactory *v_factory = nullptr;

    bool lazy_write_enabled;
    tbb::concurrent_hash_map <const TupleRow *, const TupleRow *, HashCompare> *dirty_blocks = nullptr;

    uint32_t max_calls;
    std::atomic<uint32_t> ncallbacks; // In flight write requests to the cassandra driver (not finished)

    const TableMetadata *table_metadata = nullptr;

    bool disable_timestamps;
    TimestampGenerator *timestamp_gen = nullptr;


    void flush_dirty_blocks();
    void queue_async_query(const TupleRow* keys, const TupleRow* values);

    // StorageStream attributes
    char * topic_name = nullptr;
    rd_kafka_topic_t *topic = nullptr;
    rd_kafka_t *producer = nullptr;

    std::map<std::string, std::string>* myconfig; // A reference to config, I would like to be a reference to the WriterThread, but this causes a double reference Writer<->WriterThread, this avoids the reference by forcing a call to the static method WriterThread.get(myconfig).
};


#endif //HFETCH_WRITER_H
