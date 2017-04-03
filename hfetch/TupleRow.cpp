
#include "TupleRowFactory.h"


TupleRow::TupleRow(const RowMetadata& metas,
                   uint16_t payload_size, void *buffer) {
    this->metadata = metas;
    //create data structures
    payload = std::shared_ptr<void>(buffer,
                                    [metas](void *ptr) {
                                        for (uint16_t i=0; i<metas.size(); ++i) {
                                            switch ( metas.at(i).type) {
                                                case CASS_VALUE_TYPE_BLOB:
                                                case CASS_VALUE_TYPE_TEXT:
                                                case CASS_VALUE_TYPE_VARCHAR:
                                                case CASS_VALUE_TYPE_UUID:
                                                case CASS_VALUE_TYPE_ASCII: {
                                                    int64_t *addr = (int64_t * )((char *) ptr +  metas.at(i).position);
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
    this->payload_size=t.payload_size;

    this->payload = t.payload;
    this->metadata = t.metadata;
}

TupleRow::TupleRow(const TupleRow *t) {
    this->payload_size=t->payload_size;
    this->payload = t->payload;
    this->metadata = t->metadata;
}

TupleRow::TupleRow(TupleRow &t) {
    this->payload_size=t.payload_size;
    this->payload = t.payload;
    this->metadata = t.metadata;
}

TupleRow::TupleRow(TupleRow *t) {
    this->payload_size=t->payload_size;
    this->payload = t->payload;
    this->metadata = t->metadata;
}

TupleRow &TupleRow::operator=(const TupleRow &t) {
    this->payload_size=t.payload_size;
    this->payload = t.payload;
    this->metadata = t.metadata;
    return *this;
}

TupleRow &TupleRow::operator=(TupleRow &t) {
    this->payload_size=t.payload_size;
    this->payload = t.payload;
    this->metadata = t.metadata;
    return *this;
}

bool operator<(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.payload_size != rhs.payload_size) return lhs.payload_size < rhs.payload_size;
    if (lhs.metadata.cols_meta != rhs.metadata.cols_meta) return lhs.metadata.cols_meta < rhs.metadata.cols_meta;
    return memcmp(lhs.payload.get(), rhs.payload.get(), lhs.payload_size) < 0;
}

bool operator>(const TupleRow &lhs, const TupleRow &rhs) {
    return rhs < lhs;
}

bool operator<=(const TupleRow &lhs, const TupleRow &rhs) {

    if (lhs.payload_size != rhs.payload_size) return lhs.payload_size < rhs.payload_size;
    if (lhs.metadata.cols_meta != rhs.metadata.cols_meta) return lhs.metadata.cols_meta < rhs.metadata.cols_meta;
    return memcmp(lhs.payload.get(), rhs.payload.get(), lhs.payload_size) <= 0;
}

bool operator>=(const TupleRow &lhs, const TupleRow &rhs) {
    return rhs <= lhs;
}

bool operator==(const TupleRow &lhs, const TupleRow &rhs) {

    if (lhs.payload_size != rhs.payload_size) return lhs.payload_size < rhs.payload_size;
    if (lhs.metadata.cols_meta != rhs.metadata.cols_meta) return lhs.metadata.cols_meta < rhs.metadata.cols_meta;
    return memcmp(lhs.payload.get(), rhs.payload.get(), lhs.payload_size) == 0;
}
