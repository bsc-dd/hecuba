#ifndef PREFETCHER_PREFETCHER_H
#define PREFETCHER_PREFETCHER_H
#define  TBB_USE_EXCEPTIONS 1

#include <thread>
#include <atomic>

#include "tbb/concurrent_queue.h"
#include "TupleRowFactory.h"
#include "ModuleException.h"
#include <map>

class Prefetch {

public:

    Prefetch(const std::vector<std::pair<int64_t, int64_t>> &token_ranges, const TableMetadata *table_meta,
             CassSession *session, std::map<std::string, std::string> &config);

    ~Prefetch();

    TupleRow *get_cnext();


    const TableMetadata *get_metadata() {
        return table_metadata;
    }

    inline std::string get_type() {
        return this->type;
    }

private:

    void consume_tokens();

/** no ownership **/
    CassSession *session;
    TupleRowFactory t_factory;
    std::atomic<bool> completed;

/** ownership **/

    const TableMetadata *table_metadata;
    std::thread *worker;
    tbb::concurrent_bounded_queue<TupleRow *> data;
    std::vector<std::pair<int64_t, int64_t>> tokens;
    const CassPrepared *prepared_query;
    std::string type;

};

#endif //PREFETCHER_PREFETCHER_H
