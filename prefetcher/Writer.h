//
// Created by polsm on 14/02/17.
//

#ifndef HFETCH_WRITER_H
#define HFETCH_WRITER_H


#include <thread>
#include <atomic>
#include "tbb/concurrent_queue.h"

#include "TupleRowFactory.h"

class Writer {
public:
    Writer(uint16_t buff_size, uint16_t max_callbacks, TupleRowFactory *key_factory, TupleRowFactory *value_factory, CassSession *session,
            std::string query);

    ~Writer();

    void call_async();

    void flush_elements();

    void write_to_cassandra(TupleRow* keys, TupleRow* values);

private:

    void bind(CassStatement *statement,TupleRow *tuple_row , TupleRowFactory* factory, uint16_t offset);

    CassSession* session;

    TupleRowFactory *k_factory;
    TupleRowFactory *v_factory;

    uint16_t max_calls;


/** ownership **/
    const CassPrepared *prepared_query;
    tbb::concurrent_bounded_queue<std::pair<TupleRow*,TupleRow*>> data;
    std::atomic <uint16_t > ncallbacks;
};

static void callback(CassFuture *future, void *ptr);

#endif //HFETCH_WRITER_H
