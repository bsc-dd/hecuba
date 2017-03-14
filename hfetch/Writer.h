#ifndef HFETCH_WRITER_H
#define HFETCH_WRITER_H

#define MAX_ERRORS 10

#include <thread>
#include <atomic>

#include "tbb/concurrent_queue.h"

#include "TupleRowFactory.h"

class Writer {
public:
    Writer(uint16_t buff_size, uint16_t max_callbacks, const TupleRowFactory &key_factory,
           const TupleRowFactory &value_factory, CassSession *session,
           std::string query);

    ~Writer();

    void call_async();

    void flush_elements();

    void write_to_cassandra(const TupleRow *keys, const TupleRow *values);

    void set_error_occurred(std::string error, const void * keys, const void * values);

private:

    CassSession *session;

/** ownership **/

    const CassPrepared *prepared_query;

    TupleRowFactory k_factory;
    TupleRowFactory v_factory;

    tbb::concurrent_bounded_queue<std::pair<const TupleRow *, const TupleRow *>> data;

    uint16_t max_calls;
    std::atomic<uint16_t> ncallbacks;
    std::atomic<uint16_t> error_count;

    static void callback(CassFuture *future, void *ptr);
};


#endif //HFETCH_WRITER_H
