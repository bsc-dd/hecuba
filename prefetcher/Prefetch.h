//
// Created by bscuser on 2/1/17.
//

#ifndef PREFETCHER_PREFETCHER_H
#define PREFETCHER_PREFETCHER_H


#include <atomic>
#include <thread>

#include "tbb/concurrent_queue.h"
#include "TupleRow.h"
#include "CacheTable.h"
#include "TupleRowFactory.h"
class Prefetch {
private:

/** no ownership **/
    CassSession* session;
    TupleRowFactory *t_factory;
    CacheTable *cache;

/** ownership **/
    std::thread *worker;
    tbb::concurrent_bounded_queue<TupleRow*> data;
    const std::pair<uint64_t, uint64_t> *tokens;
    const CassPrepared *prepared_query;
    uint16_t n_tokens;

protected:


public:
    //unsigned long long tokens_list[][2]
    Prefetch(const std::pair<uint64_t,uint64_t> *tokens, uint32_t buff_size, CacheTable* cache_table, CassSession* session, const char* query, uint16_t n_ranges);

    void consume_tokens();


    inline PyObject* get_next(){
        TupleRow *response;
        data.pop(response);
        if (!response) {
            data.push(NULL);
            Py_RETURN_NONE;
        }
        return t_factory->tuple_as_py(response);
    }

    inline TupleRow* get_next_tuple(){
        TupleRow *response;
        data.pop(response);
        return response;
    }

    ~Prefetch() {
        worker->join();
        cass_prepared_free(this->prepared_query);
        data.clear();
    }


};
/*
#include <python2.7/Python.h>
#include <cassandra.h>
#include <atomic>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <cstring>

#include <vector>
#include <string>
#include <unordered_map>



class TokenGenerator{
public:

    TokenGenerator() {

    }
    TokenGenerator(unsigned long long tokens_list[][2]) {
        this->count = 0;
        int n_ranges =(sizeof(*tokens_list) / (sizeof(unsigned long long)*2) );
        this->tokens = std::vector <std::vector <unsigned long long> >(n_ranges,std::vector<unsigned long long>(2));

        int i = 0;
        while (i<n_ranges) {
            this->tokens[i][0]=tokens_list[i][0];
            this->tokens[i][1]=tokens_list[i][1];
            ++i;
        }
    };

    ~TokenGenerator() {}

    std::vector<unsigned long long> get_next() {
        count++;
        if (count-1==tokens.size()) return std::vector<unsigned long long>();
        return tokens[count-1];
    }

private:
    std::vector <std::vector <unsigned long long> > tokens;
    std::atomic <unsigned int> count;
};

}
*/


#endif //PREFETCHER_PREFETCHER_H
