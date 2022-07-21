#include "TupleRow.h"
#include "UUID.h"


TupleRow::TupleRow(std::shared_ptr<const std::vector<ColumnMeta>> metas,
                   size_t payload_size, void *buffer) {

    metadatas = metas;
    payload = std::shared_ptr<TupleRowData>(new TupleRowData(buffer, payload_size, (uint32_t) metas->size()),
                                            [metas](TupleRowData *holder) {
                                                for (uint16_t i = 0; i < metas->size(); ++i) {
                                                    if (!holder->isNull(i)) {
                                                        switch (metas->at(i).type) {
                                                            case CASS_VALUE_TYPE_BLOB:
                                                            case CASS_VALUE_TYPE_TEXT:
                                                            case CASS_VALUE_TYPE_VARCHAR:
                                                            case CASS_VALUE_TYPE_ASCII: {
                                                                int64_t *addr = (int64_t *) ((char *) holder->data +
                                                                                             metas->at(i).position);
                                                                char *d = reinterpret_cast<char *>(*addr);
                                                                free(d);
                                                                break;
                                                            }
                                                            case CASS_VALUE_TYPE_UUID: {
                                                                int64_t *addr = (int64_t *) ((char *) holder->data +
                                                                                             metas->at(i).position);
                                                                uint64_t *uuid = reinterpret_cast<uint64_t *>(*addr);
                                                                delete[] uuid; //#TODO: Check interaction with hdict interface
                                                                break;
                                                            }
                                                            case CASS_VALUE_TYPE_TUPLE: {
                                                                int64_t *addr = (int64_t *) ((char *) holder->data +
                                                                                             metas->at(i).position);
                                                                TupleRow *tr = reinterpret_cast<TupleRow *>(*addr);
                                                                delete (tr);
                                                                break;
                                                            }
                                                            case CASS_VALUE_TYPE_UDT: {
                                                                int64_t *addr = (int64_t *) ((char *) holder->data +
                                                                                             metas->at(i).position);
                                                                char *udt = reinterpret_cast<char *>(*addr);
                                                                free(udt);
                                                            }
                                                            default:
                                                                break;
                                                        }
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

std::string TupleRow::show_content(void) const {
    std::string result;
    char * addr = (char*)this->get_payload();
    std::string tmp;

    for (uint16_t i = 0; i < metadatas->size(); ++i) {
        if (!isNull(i)) {
            switch (metadatas->at(i).type) {
                case CASS_VALUE_TYPE_VARCHAR:
                case CASS_VALUE_TYPE_TEXT:
                case CASS_VALUE_TYPE_ASCII:
                {
                    tmp = std::string(*(char**)addr);
                    break;
                }
                case CASS_VALUE_TYPE_DATE:
                case CASS_VALUE_TYPE_TIME:
                case CASS_VALUE_TYPE_TIMESTAMP:
                {
                    tmp = std::string("DATE(todo)");
                    break;
                }
                case CASS_VALUE_TYPE_UUID:{
                    tmp = UUID::UUID2str(*(uint64_t**)addr);
                    break;
                }
                case CASS_VALUE_TYPE_TUPLE:{
                    /* we will allocate the size of the inner tupple */
                    tmp = std::string("TUPLE(todo)");
                    break;
                }
                case CASS_VALUE_TYPE_UDT:
                case CASS_VALUE_TYPE_BLOB: {
                    // first the size and then the content of the blob or the numpy
                    tmp = std::string("NUMPYMETAS or NUMPY(todo)");
                    break;
                }
                case CASS_VALUE_TYPE_VARINT:{
                    tmp = std::string("VARINT(todo)");
                    break;
                }
                case CASS_VALUE_TYPE_BIGINT:{
                    tmp = std::to_string(*(uint64_t*)addr);
                    break;
                }
                case CASS_VALUE_TYPE_BOOLEAN:{
                    tmp = std::to_string(*(bool*)addr);
                    break;
                }
                case CASS_VALUE_TYPE_COUNTER:{
                    tmp = std::string("COUNTER(todo)");
                    break;
                }
                case CASS_VALUE_TYPE_DOUBLE:{
                    tmp = std::to_string(*(double*)addr);
                    break;
                }
                case CASS_VALUE_TYPE_FLOAT:{
                    tmp = std::to_string(*(float*)addr);
                    break;
                }
                case CASS_VALUE_TYPE_INT:{
                    tmp = std::to_string(*(uint32_t*)addr);
                    break;
                }
                case CASS_VALUE_TYPE_SMALL_INT: {
                    tmp = std::to_string(*(uint16_t*)addr);
                    break;
                }
                case CASS_VALUE_TYPE_TINY_INT: {
                    tmp = std::to_string(*(uint8_t*)addr);
                    break;
                }
                case CASS_VALUE_TYPE_DECIMAL:
                case CASS_VALUE_TYPE_TIMEUUID:
                case CASS_VALUE_TYPE_INET:
                case CASS_VALUE_TYPE_LIST:
                case CASS_VALUE_TYPE_MAP:
                case CASS_VALUE_TYPE_SET:
                case CASS_VALUE_TYPE_CUSTOM:
                case CASS_VALUE_TYPE_UNKNOWN:
                default: {
                    tmp = std::string("UNKNOWN type [") + std::to_string(metadatas->at(i).type) + std::string("]");
                    break;
                }
            }
            addr += metadatas->at(i).size;
            result += tmp + " ";
        }

    }
    return result;
}
