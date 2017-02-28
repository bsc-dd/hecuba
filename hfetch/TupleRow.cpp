
#include "TupleRowFactory.h"


TupleRow::TupleRow(const std::shared_ptr <std::vector<ColumnMeta>> metas,
                   uint16_t payload_size, void *buffer) {
    this->metadata = metas;
    //create data structures
    payload = std::shared_ptr<void>(buffer,
                                    [metas](void *ptr) {

                                        for (auto &m : *metas.get()) {
                                            switch (m.type) {
                                                case CASS_VALUE_TYPE_TEXT:
                                                case CASS_VALUE_TYPE_VARCHAR:
                                                case CASS_VALUE_TYPE_ASCII: {
                                                    int64_t *addr = (int64_t * )((char *) ptr + m.position);
                                                    char *d = reinterpret_cast<char *>(*addr);
                                                    free(d);
                                                    break;
                                                }
                                                default:
                                                    break;
                                            }
                                        }
                                        free(ptr);

                                    });

    this->payload_size = payload_size;
}


TupleRow::TupleRow(const TupleRow &t) {
    this->payload = t.payload;
    this->metadata = t.metadata;
}

TupleRow::TupleRow(const TupleRow *t) {
    this->payload = t->payload;
    this->metadata = t->metadata;
}

TupleRow::TupleRow(TupleRow &t) {
    this->payload = t.payload;
    this->metadata = t.metadata;
}

TupleRow::TupleRow(TupleRow *t) {
    this->payload = t->payload;
    this->metadata = t->metadata;
}

TupleRow &TupleRow::operator=(const TupleRow &t) {
    this->payload = t.payload;
    this->metadata = t.metadata;
    return *this;
}

TupleRow &TupleRow::operator=(TupleRow &t) {
    this->payload = t.payload;
    this->metadata = t.metadata;
    return *this;
}

bool operator<(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.payload_size != rhs.payload_size || lhs.metadata != rhs.metadata) {
        return lhs.payload_size < rhs.payload_size;
    } else {
        return memcmp(lhs.payload.get(), rhs.payload.get(), lhs.payload_size) < 0;
    }
}

bool operator>(const TupleRow &lhs, const TupleRow &rhs) {
    return rhs < lhs;
}

bool operator<=(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.payload_size != rhs.payload_size || lhs.metadata != rhs.metadata) {
        return lhs.payload_size < rhs.payload_size;
    } else {
        return memcmp(lhs.payload.get(), rhs.payload.get(), lhs.payload_size) <= 0;
    }
}

bool operator>=(const TupleRow &lhs, const TupleRow &rhs) {
    return rhs <= lhs;
}

bool operator==(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.payload_size != rhs.payload_size || lhs.metadata != rhs.metadata) return false;
    return memcmp(lhs.payload.get(), rhs.payload.get(), lhs.payload_size) == 0;
}
