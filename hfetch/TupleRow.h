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

public:

    TupleRow(const std::vector<uint16_t> *size_elem, uint16_t payload_size,void *free);

    ~TupleRow();

    TupleRow(const TupleRow &t) ;

    TupleRow(const TupleRow *t);

    inline const uint16_t n_elem() const {
        if (!positions || (*positions).empty()) return 0;
        return (uint16_t) positions->size();
    }

    const void* get_element(int32_t position) const {
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
