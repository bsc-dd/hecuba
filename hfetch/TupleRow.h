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
    struct TupleRowData {
        void* data;
        uint32_t null_values;
        uint32_t length;

        TupleRowData(void* data_ptr, uint32_t length) {
            this->data=data_ptr;
            this->null_values=0;
            this->length = length;
        }

        ~TupleRowData() {
            free(data);
        }

        bool operator<(TupleRowData &rhs) {
            if (this->length != rhs.length) return this->length < rhs.length;
            if (this->null_values!=rhs.null_values)
                return this->null_values < rhs.null_values;
            return memcmp(this->data, rhs.data, this->length) < 0;
        }

        bool operator>(TupleRowData &rhs) {
            return rhs < *this;
        }

        bool operator<=(TupleRowData &rhs) {
            if (this->length != rhs.length) return this->length < rhs.length;
            if (this->null_values!=rhs.null_values)
                return this->null_values < rhs.null_values;
            return memcmp(this->data, rhs.data, this->length) <= 0;
        }

        bool operator>=(TupleRowData &rhs) {
            return rhs <= *this;
        }

        bool operator==(TupleRowData &rhs) {
            if (this->length != rhs.length) return this->length < rhs.length;
            if (this->null_values!=rhs.null_values)
                return this->null_values < rhs.null_values;
            return memcmp(this->data, rhs.data, length) == 0;
        }

    };



    std::shared_ptr<TupleRowData> payload;
    std::shared_ptr<const std::vector<ColumnMeta>> metadatas;

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


    /* Set methods */
//TODO implement this ops with atomic_exchange
    void setNull(uint32_t position) {
        this->payload->null_values= this->payload->null_values|(0x1<<position);
    }

    void unsetNull(uint32_t position) {
        this->payload->null_values= this->payload->null_values &!(0x1<<position);
    }

    /* Get methods */
    bool isNull(uint32_t position) const {
        return  this->payload->null_values&(0x1<<position);
    }

    inline void* get_payload() const{
        return this->payload->data;
    }

    inline const uint16_t n_elem() const {
        return (uint16_t) metadatas->size();
    }

    inline const void* get_element(int32_t position) const {
        if (position < 0 || payload->data == nullptr) return nullptr;
        if (isNull((uint32_t)position)) return nullptr;
        return (const char *) payload->data + metadatas->at(position).position;
    }

    /* Comparision operators */

    friend bool operator<(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator<=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator==(const TupleRow &lhs, const TupleRow &rhs);


};

#endif //PREFETCHER_MY_TUPLE_H
