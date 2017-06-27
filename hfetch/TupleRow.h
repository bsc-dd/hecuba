#ifndef TUPLEROW_H
#define TUPLEROW_H

#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <stdlib.h>
#include <vector>
#include <cassandra.h>


#include "TableMetadata.h"


class TupleRow {

public:

    /* Constructor */
    TupleRow(std::shared_ptr<const std::vector<ColumnMeta> > metas, uint32_t payload_size, void *buffer);

    /* Copy constructors */
    TupleRow(const TupleRow &t);

    TupleRow(const TupleRow *t);

    TupleRow(TupleRow *t);

    TupleRow(TupleRow &t);

    TupleRow &operator=(const TupleRow &other);

    TupleRow &operator=(TupleRow &other);


    /* Set methods */
    inline void setNull(uint32_t position) {
        this->payload->setNull(position);
    }

    inline void unsetNull(uint32_t position) {
        this->payload->unsetNull(position);
    }

    /* Get methods */
    inline bool isNull(uint32_t position) const {
        return this->payload->isNull(position);
    }

    inline void *get_payload() const {
        return this->payload->data;
    }

    inline const uint16_t n_elem() const {
        return (uint16_t) metadatas->size();
    }

    inline const void *get_element(uint32_t position) const {
        if (!isNull(position))
            return (char *) payload->data + metadatas->at(position).position;
        return nullptr;
    }

    /* Comparision operators */
    friend bool operator<(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator<=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator>=(const TupleRow &lhs, const TupleRow &rhs);

    friend bool operator==(const TupleRow &lhs, const TupleRow &rhs);


private:

    struct TupleRowData {

        /* Attributes */
        void *data;
        uint32_t null_values, length;


        /* Constructors */
        TupleRowData(void *data_ptr, uint32_t length) {
            this->data = data_ptr;
            this->null_values = 0;
            this->length = length;
        }

        /* Destructors */
        ~TupleRowData() {
            free(data);
        }

        /* Modifiers */
        void setNull(uint32_t position) {
            this->null_values |= (0x1 << position);
        }

        void unsetNull(uint32_t position) {
            this->null_values &= !(0x1 << position);
        }

        /* Get methods */
        bool isNull(uint32_t position) const {
            return data == nullptr || (this->null_values & (0x1 << position)) > 0;
        }


        /* Comparators */
        bool operator<(TupleRowData &rhs) {
            if (this->length != rhs.length) return this->length < rhs.length;
            if (this->null_values != rhs.null_values)
                return this->null_values < rhs.null_values;
            return memcmp(this->data, rhs.data, this->length) < 0;
        }

        bool operator>(TupleRowData &rhs) {
            return rhs < *this;
        }

        bool operator<=(TupleRowData &rhs) {
            if (this->length != rhs.length) return this->length < rhs.length;
            if (this->null_values != rhs.null_values)
                return this->null_values < rhs.null_values;
            return memcmp(this->data, rhs.data, this->length) <= 0;
        }

        bool operator>=(TupleRowData &rhs) {
            return rhs <= *this;
        }

        bool operator==(TupleRowData &rhs) {
            if (this->length != rhs.length) return this->length < rhs.length;
            if (this->null_values != rhs.null_values)
                return this->null_values < rhs.null_values;
            return memcmp(this->data, rhs.data, length) == 0;
        }

    };


    std::shared_ptr<TupleRowData> payload;
    std::shared_ptr<const std::vector<ColumnMeta>> metadatas;
};

#endif //TUPLEROW_H
