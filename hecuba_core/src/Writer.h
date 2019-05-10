#ifndef HFETCH_WRITER_H
#define HFETCH_WRITER_H

#define MAX_ERRORS 10

#include <thread>
#include <atomic>
#include <map>

#include "tbb/concurrent_queue.h"

#include "TimestampGenerator.h"
#include "TupleRowFactory.h"


class Writer {
public:
    Writer(const TableMetadata *table_meta, CassSession *session,
           std::map<std::string, std::string> &config);

    ~Writer();

    void set_timestamp_gen(TimestampGenerator &time_gen);

    void call_async();

    void flush_elements();

    void write_to_cassandra(const TupleRow *keys, const TupleRow *values);

    void write_to_cassandra(void *keys, void *values);

    void set_error_occurred(std::string error, const void *keys, const void *values);

    const TableMetadata *get_metadata() {
        return table_metadata;
    }

private:

    CassSession *session;

/** ownership **/

    const CassPrepared *prepared_query;

    TupleRowFactory *k_factory;
    TupleRowFactory *v_factory;

    tbb::concurrent_bounded_queue<std::pair<const TupleRow *, const TupleRow *>> data;

    uint32_t max_calls;
    std::atomic<uint32_t> ncallbacks;
    std::atomic<uint32_t> error_count;
    const TableMetadata *table_metadata;

    TimestampGenerator timestamp_gen;

    static void callback(CassFuture *future, void *ptr);
};


#endif //HFETCH_WRITER_H
