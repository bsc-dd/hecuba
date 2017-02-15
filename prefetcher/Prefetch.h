//
// Created by bscuser on 2/1/17.
//

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

    void consume_tokens();



    PyObject* get_next();

    inline TupleRow* get_next_tuple(){
        TupleRow *response;
        try {
            data.pop(response);
        }
        catch (tbb::user_abort& e) {

            data.set_capacity(0);
        }
        session=NULL;
        return response;
    }

    ~Prefetch() {

        //clear calls items destructors
        data.clear();

        worker->join();
        delete(worker);

        cass_prepared_free(this->prepared_query);
    }

private:

/** no ownership **/
    CassSession* session;
    TupleRowFactory *t_factory;
    bool completed;

/** ownership **/
    std::thread* worker;
    tbb::concurrent_bounded_queue<TupleRow*> data;
    const std::vector<std::pair<int64_t, int64_t>> *tokens;
    const CassPrepared *prepared_query;

protected:

};

#endif //PREFETCHER_PREFETCHER_H
