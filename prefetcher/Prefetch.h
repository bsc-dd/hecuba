#ifndef PREFETCHER_PREFETCHER_H
#define PREFETCHER_PREFETCHER_H
#define  TBB_USE_EXCEPTIONS 1

#include <thread>

#include "tbb/concurrent_queue.h"

#include "TupleRowFactory.h"
#include "ModuleException.h"

class Prefetch {

public:

    Prefetch(const std::vector<std::pair<int64_t, int64_t>> *tokens, uint32_t buff_size, TupleRowFactory* tuple_factory,
             CassSession* session,std::string query);

    ~Prefetch();

    PyObject* get_next();

private:

    void consume_tokens();

/** no ownership **/
    CassSession* session;
    TupleRowFactory *t_factory;
    bool completed;
    const char *error_msg;

/** ownership **/
    std::thread* worker;
    tbb::concurrent_bounded_queue<TupleRow*> data;
    const std::vector<std::pair<int64_t, int64_t>> *tokens;
    const CassPrepared *prepared_query;

    TupleRow *get_cnext();
};

#endif //PREFETCHER_PREFETCHER_H
