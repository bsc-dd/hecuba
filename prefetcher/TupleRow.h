#ifndef PREFETCHER_MY_TUPLE_H
#define PREFETCHER_MY_TUPLE_H

#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <iostream>
#include "stdlib.h"
#include <vector>
#include <cassandra.h>
#include <python2.7/Python.h>


class TupleRow {
private:
    std::shared_ptr<const void> payload;
    const std::vector<uint16_t> *positions;
    uint16_t payload_size;
/*

    inline void *get_element(int32_t position) const {
        if (position < 0 || payload.get() == 0) return 0;
        return (char *) payload.get() + (*positions)[position];
    }
*/
public:

    TupleRow(const std::vector<uint16_t> *size_elem, uint16_t payload_size,void *free);

    ~TupleRow() {
        //payload is freed by shared pointer deleter
        payload_size = 0;
        positions = 0;
    }

    //this shouldn't be used
    TupleRow(const TupleRow &t) {
        //  throw std::runtime_error("Copy constructor called");
        this->payload = t.payload;
        this->payload_size = t.payload_size;
        this->positions = t.positions;
    }

    inline uint16_t n_elem() {
        if (!positions || (*positions).empty()) return 0;
        return (uint16_t) positions->size();
    }

    const void* get_element(int32_t position) {
        if (position < 0 || payload.get() == 0) return 0;
        return (const char *) payload.get() + (*positions)[position];
    }

    friend bool operator<(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator<=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator==(const TupleRow &lhs, const TupleRow &rhs);


};

#endif //PREFETCHER_MY_TUPLE_H
