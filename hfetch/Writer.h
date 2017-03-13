//
// Created by polsm on 14/02/17.
//

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

    void set_error(std::string error,const TupleRow* keys, const TupleRow* values);

    struct currentUnit{
        Writer *W;
        const TupleRow* key, *value;
        currentUnit(Writer* W, const TupleRow* k, const TupleRow* v ):W(W), key(k), value(v) {}

    };
private:



    CassSession *session;

    TupleRowFactory k_factory;
    TupleRowFactory v_factory;

    uint16_t max_calls;

    std::atomic<uint16_t> error_count;


/** ownership **/
    const CassPrepared *prepared_query;
    tbb::concurrent_bounded_queue<std::pair<const TupleRow *, const TupleRow *>> data;
    std::atomic<uint16_t> ncallbacks;

};

static void callback(CassFuture *future, void *ptr);

#endif //HFETCH_WRITER_H
