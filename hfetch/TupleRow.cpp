#include "TupleRow.h"


TupleRow::TupleRow(std::shared_ptr<const std::vector<ColumnMeta>> metas,
                   uint32_t payload_size, void *buffer) {

    metadatas = metas;
    payload = std::shared_ptr<TupleRowData>(new TupleRowData(buffer, payload_size),
                                            [metas](TupleRowData *holder) {
                                                for (uint16_t i = 0; i < metas->size(); ++i) {
                                                    switch (metas->at(i).type) {
                                                        case CASS_VALUE_TYPE_BLOB:
                                                        case CASS_VALUE_TYPE_TEXT:
                                                        case CASS_VALUE_TYPE_VARCHAR:
                                                        case CASS_VALUE_TYPE_UUID:
                                                        case CASS_VALUE_TYPE_ASCII: {
                                                            if (!holder->isNull(i)) {
                                                                int64_t *addr = (int64_t *) ((char *) holder->data +
                                                                                             metas->at(i).position);
                                                                char *d = reinterpret_cast<char *>(*addr);
                                                                free(d);
                                                            }
                                                            break;
                                                        }
                                                        default:
                                                            break;
                                                    }
                                                }
                                                delete (holder);
                                            });
}


TupleRow::TupleRow(const TupleRow &t) {
    this->metadatas = t.metadatas;
    this->payload = t.payload;
}

TupleRow::TupleRow(const TupleRow *t) {
    this->metadatas = t->metadatas;
    this->payload = t->payload;
}

TupleRow::TupleRow(TupleRow &t) {
    this->metadatas = t.metadatas;
    this->payload = t.payload;
}

TupleRow::TupleRow(TupleRow *t) {
    this->metadatas = t->metadatas;
    this->payload = t->payload;
}

TupleRow &TupleRow::operator=(const TupleRow &t) {
    this->metadatas = t.metadatas;
    this->payload = t.payload;
    return *this;
}

TupleRow &TupleRow::operator=(TupleRow &t) {
    this->metadatas = t.metadatas;
    this->payload = t.payload;
    return *this;
}

bool operator<(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.metadatas != rhs.metadatas) return lhs.metadatas < rhs.metadatas;
    return *lhs.payload.get() < *rhs.payload.get();
}

bool operator>(const TupleRow &lhs, const TupleRow &rhs) {
    return rhs < lhs;
}

bool operator<=(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.metadatas != rhs.metadatas) return lhs.metadatas < rhs.metadatas;
    return *lhs.payload.get() <= *rhs.payload.get();
}

bool operator>=(const TupleRow &lhs, const TupleRow &rhs) {
    return rhs <= lhs;
}

bool operator==(const TupleRow &lhs, const TupleRow &rhs) {
    if (lhs.metadatas != rhs.metadatas) return false;
    return *lhs.payload.get() == *rhs.payload.get();
}