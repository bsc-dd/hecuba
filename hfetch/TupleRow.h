#ifndef PREFETCHER_MY_TUPLE_H
#define PREFETCHER_MY_TUPLE_H

#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <stdlib.h>
#include <vector>
#include <cassandra.h>


#include "TableMetadata.h"


class TupleRow {
private:
    std::shared_ptr<void> payload;
    std::shared_ptr<const std::vector<ColumnMeta> > metadata;
    uint16_t payload_size;
public:

    /* Constructor */
    TupleRow(std::shared_ptr<const std::vector<ColumnMeta>> metas, uint16_t payload_size,void *buffer);


    /* Copy constructors */
    TupleRow(const TupleRow &t) ;

    TupleRow(const TupleRow *t);

    TupleRow(TupleRow *t);

    TupleRow(TupleRow &t);

    TupleRow& operator=( const TupleRow& other );

    TupleRow& operator=(TupleRow& other );

    /* Get methods */

    inline std::shared_ptr<void>  get_payload() const{
        return this->payload;
    }

inline const uint16_t get_payload_size() const {
    return this->payload_size;
}
    inline const uint16_t n_elem() const {
        return (uint16_t) metadata->size();
    }

    inline const void* get_element(int32_t position) const {
        if (position < 0 || payload.get() == 0) return 0;
        return (const char *) payload.get() + metadata->at(position).position;
    }

    /* Comparision operators */

    friend bool operator<(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator<=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator==(const TupleRow &lhs, const TupleRow &rhs);


};

#endif //PREFETCHER_MY_TUPLE_H
