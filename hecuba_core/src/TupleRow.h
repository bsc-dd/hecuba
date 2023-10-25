#ifndef TUPLEROW_H
#define TUPLEROW_H

#include <iostream>
#include <cstring>
#include <string>
#include <stdlib.h>
#include <vector>
#include <cassandra.h>
#include <cmath>


#include "TableMetadata.h"


class TupleRow {

public:

    /* Constructor */
    TupleRow(std::shared_ptr<const std::vector<ColumnMeta> > metas, size_t payload_size, void *buffer);

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
    inline void set_null_values(std::vector<uint32_t> v) {
        this->payload->null_values=v;
    }

    inline void set_timestamp(int64_t timestamp) {
        this->payload->setTimestamp(timestamp);
    }


    /* Get methods */
    inline bool isNull(uint32_t position) const {
        return this->payload->isNull(position);
    }

    inline uint64_t get_timestamp() const {
        return this->payload->getTimestamp();
    }

    inline void *get_payload() const {
        return this->payload->data;
    }
    inline std::vector<uint32_t>const& get_null_values() const {
        return this->payload->null_values;
    }

    inline size_t length() const {
        return this->payload->ptr_length;
    }

    inline const uint16_t n_elem() const {
        return (uint16_t) metadatas->size();
    }

    inline const ColumnMeta &get_metadata_element(uint32_t position) const {
        return metadatas->at(position);
    }

    inline const void *get_element(uint32_t position) const {
        if (!isNull(position))
            return (char *) payload->data + metadatas->at(position).position;
        return nullptr;
    }

    inline int64_t use_count() const {
        return this->payload.use_count();
    }

    std::string show_content() const;

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
        size_t ptr_length;
        std::vector<uint32_t> null_values;
        int64_t timestamp;


        /* Constructors */
        TupleRowData(void *data_ptr, size_t length, uint32_t nelem) {
            this->data = data_ptr;
            this->null_values = std::vector<uint32_t>(ceil((double) nelem/32), 0);
            this->ptr_length = length;
            this->timestamp = 0;
        }

        /* Destructors */
        ~TupleRowData() {
            free(data);
        }

        /* Modifiers */
        /*
         * Every position of the null values vector represents 32 values
         * of the data payload. Therefore, to decide which bucket inside the vector
         * we need to access the position must be divided by 32. This is accomplished by
         * doing bit shifting (5 positions to the right since 2^5=32).
         */
        void setNull(uint32_t position) {
            if (!null_values.empty()) this->null_values[position >> 5] |= (0x1 << (position % 32));
        }

        void unsetNull(uint32_t position) {
            if (!null_values.empty()) this->null_values[position >> 5] &= ~(0x1 << (position % 32));
        }

        void setTimestamp(int64_t timestamp) {
            this->timestamp = timestamp;
        }

        /* Get methods */
        bool isNull(uint32_t position) const {
            if (!data || null_values.empty()) return true;
            return (this->null_values[position >> 5] & (0x1 << position % 32)) > 0;
        }

        uint64_t getTimestamp() const {
            return this->timestamp;
        }


        /* Comparators */
        bool operator<(TupleRowData &rhs) {
            if (this->ptr_length != rhs.ptr_length) return this->ptr_length < rhs.ptr_length;
            if (this->null_values.size() != rhs.null_values.size())
                return this->null_values < rhs.null_values;
            if (this->null_values != rhs.null_values)
                return this->null_values < rhs.null_values;
            return memcmp(this->data, rhs.data, this->ptr_length) < 0;
        }

        bool operator>(TupleRowData &rhs) {
            return rhs < *this;
        }

        bool operator<=(TupleRowData &rhs) {
            if (this->ptr_length != rhs.ptr_length) return this->ptr_length < rhs.ptr_length;
            if (this->null_values.size() != rhs.null_values.size())
                return this->null_values < rhs.null_values;
            if (this->null_values != rhs.null_values)
                return this->null_values < rhs.null_values;
            return memcmp(this->data, rhs.data, this->ptr_length) <= 0;
        }

        bool operator>=(TupleRowData &rhs) {
            return rhs <= *this;
        }

        bool operator==(TupleRowData &rhs) {
            if (this->ptr_length != rhs.ptr_length) return false;
            if (this->null_values.size() != rhs.null_values.size())
                return false;
            if (this->null_values != rhs.null_values)
                return false;
            return memcmp(this->data, rhs.data, ptr_length) == 0;
        }

    };


    std::shared_ptr<TupleRowData> payload;
    std::shared_ptr<const std::vector<ColumnMeta>> metadatas;
};


// Allows indexing TupleRows in a hash map. Trivial implementation, a custom hash should reduce overhead.
namespace std {
    template<>
    struct hash<TupleRow> {
        std::size_t operator()(const TupleRow &k) const {
            return hash<std::string>()(std::string((char *) k.get_payload(), k.length()));
        }
    };
}


#endif //TUPLEROW_H
