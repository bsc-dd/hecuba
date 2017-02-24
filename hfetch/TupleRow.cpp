
#include "TupleRowFactory.h"


TupleRow::TupleRow(const std::vector<uint16_t> *size_elem, uint16_t payload_size,   void *buffer) {
    //create data structures
    payload = std::shared_ptr<void>(buffer, free);
    positions = size_elem;
    this->payload_size = payload_size;
}

TupleRow::~TupleRow() {
    //payload is freed by shared pointer deleter
    payload_size = 0;
    positions = 0;
}

TupleRow::TupleRow(const TupleRow &t) {
    this->payload = t.payload;
    this->payload_size = t.payload_size;
    this->positions = t.positions;
}

TupleRow::TupleRow(const TupleRow *t) {
    this->payload = t->payload;
    this->payload_size = t->payload_size;
    this->positions = t->positions;
}

bool operator<(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.payload_size != rhs.payload_size) {
        return lhs.payload_size < rhs.payload_size;
    } else {
        return memcmp(lhs.payload.get(), rhs.payload.get(), lhs.payload_size) < 0;
    }
}

bool operator>(const TupleRow &lhs, const TupleRow &rhs) {
    return rhs < lhs;
}

bool operator<=(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.payload_size != rhs.payload_size) {
        return lhs.payload_size < rhs.payload_size;
    } else {
        return memcmp(lhs.payload.get(), rhs.payload.get(), lhs.payload_size) <= 0;
    }
}

bool operator>=(const TupleRow &lhs, const TupleRow &rhs) {
    return rhs <= lhs;
}

bool operator==(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.payload_size != rhs.payload_size) return false;
    return memcmp(lhs.payload.get(), rhs.payload.get(), lhs.payload_size) == 0;
}
