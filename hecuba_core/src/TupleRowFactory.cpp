#include "TupleRowFactory.h"

/***
 * Builds a tuple factory to retrieve tuples based on rows and keys
 * extracting the information from Cassandra to decide the types to be used
 * @param table_meta Holds the table information
 */
TupleRowFactory::TupleRowFactory(std::shared_ptr<const std::vector<ColumnMeta> > row_info) {
    this->metadata = row_info;
    this->total_bytes = 0;
    if (row_info->end() != row_info->begin()) {
        std::vector<ColumnMeta>::const_iterator last_element  = --row_info->end();
        total_bytes = last_element->position + last_element->size;
    }
}



/*** TUPLE BUILDERS ***/

/***
 * Build a tuple taking the given buffer as its internal representation
 * @param data Valid pointer to the values
 * @return TupleRow with the buffer as its inner data
 * @post The TupleRow now owns the data and this cannot be freed
 */
TupleRow *TupleRowFactory::make_tuple(void *data) {
    return new TupleRow(metadata, total_bytes, data);
}

/***
 * Build a tuple from the given Cassandra result row using the factory's metadata
 * @param row Contains the same number of columns than the metadata
 * @return TupleRow with a copy of the values inside the row
 * @post The row can be freed
 */
TupleRow *TupleRowFactory::make_tuple(const CassRow *row) {
    if (!row) return NULL;
    char *buffer = nullptr;

    if (total_bytes>0) buffer = (char *) malloc(total_bytes);

    TupleRow *new_tuple = new TupleRow(metadata, total_bytes, buffer);

    CassIterator *it = cass_iterator_from_row(row);
    for (uint16_t i = 0; cass_iterator_next(it) && i<metadata->size(); ++i) {
        if (cass_to_c(cass_iterator_get_column(it), buffer + metadata->at(i).position, i) == -1) {
            new_tuple->setNull(i);
        }
    }
    cass_iterator_free(it);
    return new_tuple;
}


/***
 * @pre: -
 * @post: Extract the Cassandra's value from lhs and writes it to the memory pointed by data
 * using the data type information provided by type_array[col]
 * @param lhs Cassandra value
 * @param data Pointer to the place where the extracted value "lhs" should be written
 * @param col Indicates which column, thus, which data type are we processing
 * @return 0 if succeeds
 */
int TupleRowFactory::cass_to_c(const CassValue *lhs, void *data, int16_t col) const {

    if (col < 0 || col >= (int32_t) metadata->size()) {
        throw ModuleException("TupleRowFactory: Cass to C: Asked for column " + std::to_string(col) + " but only " +
                              std::to_string(metadata->size()) + " are present");
    }
    switch (metadata->at(col).type) {
        case CASS_VALUE_TYPE_TEXT:{
            const char *l_temp;
            size_t l_size;
            CassError rc = cass_value_get_string(lhs, &l_temp, &l_size);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse text unsuccessful, column" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
            char *permanent = (char *) malloc(l_size + 1);
            memcpy(permanent, l_temp, l_size);
            permanent[l_size] = '\0';
            memcpy(data, &permanent, sizeof(char *));
            return 0;
        }
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_ASCII: {
            const char *l_temp;
            size_t l_size;
            CassError rc = cass_value_get_string(lhs, &l_temp, &l_size);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse text unsuccessful, column" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
            char *permanent = (char *) malloc(l_size + 1);
            memcpy(permanent, l_temp, l_size);
            permanent[l_size] = '\0';
            memcpy(data, &permanent, sizeof(char *));
            return 0;
        }
        case CASS_VALUE_TYPE_VARINT:
        case CASS_VALUE_TYPE_BIGINT: {
            int64_t *p = static_cast<int64_t * >(data);
            CassError rc = cass_value_get_int64(lhs, p);
            CHECK_CASS(
                    "TupleRowFactory: Cassandra to C parse bigint/varint unsuccessful, column:" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
            return 0;
        }
        case CASS_VALUE_TYPE_BLOB: {
            const unsigned char *l_temp;
            size_t l_size;
            CassError rc = cass_value_get_bytes(lhs, &l_temp, &l_size);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse bytes unsuccessful, column:" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;

            //Allocate space for the bytes
            char *permanent = (char *) malloc(l_size + sizeof(uint64_t));
            //TODO make sure l_size < uint32 max
            uint64_t int_size = (uint64_t) l_size;

            //copy num bytes
            memcpy(permanent, &int_size, sizeof(uint64_t));

            //copy bytes
            memcpy(permanent + sizeof(uint64_t), l_temp, l_size);

            //copy pointer to payload
            memcpy(data, &permanent, sizeof(char *));
            return 0;
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
            cass_bool_t b;
            CassError rc = cass_value_get_bool(lhs, &b);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse bool unsuccessful, column:" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
            bool *p = static_cast<bool *>(data);
            if (b == cass_true) *p = true;
            else *p = false;
            return 0;
        }
        case CASS_VALUE_TYPE_COUNTER: {
            CassError rc = cass_value_get_uint32(lhs, reinterpret_cast<uint32_t *>(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse counter as uint32 unsuccessful, column:" +
                       std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
            return 0;
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //TODO
            //decimal.Decimal
            break;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            CassError rc = cass_value_get_double(lhs, reinterpret_cast<double *>(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse double unsuccessful, column:" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
            return 0;
        }
        case CASS_VALUE_TYPE_FLOAT: {
            CassError rc = cass_value_get_float(lhs, reinterpret_cast<float * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse float unsuccessful, column:" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
            return 0;
        }
        case CASS_VALUE_TYPE_INT: {
            CassError rc = cass_value_get_int32(lhs, reinterpret_cast<int32_t * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse int32 unsuccessful, column:" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
            return 0;
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_UUID: {
            CassUuid uuid;
            CassError rc = cass_value_get_uuid(lhs, &uuid);
            uint64_t time_and_version = uuid.time_and_version;
            uint64_t clock_seq_and_node = uuid.clock_seq_and_node;

            CHECK_CASS("TupleRowFactory: Cassandra to C parse UUID unsuccessful, column:" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
            char *permanent = (char *) malloc(sizeof(uint64_t) * 2);
            memcpy(permanent, &time_and_version, sizeof(uint64_t));
            memcpy(permanent + sizeof(uint64_t), &clock_seq_and_node, sizeof(uint64_t));
            memcpy(data, &permanent, sizeof(char *));
            return 0;
        }
        case CASS_VALUE_TYPE_TIMEUUID: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_INET: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_DATE: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_TIME: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_SMALL_INT: {
            CassError rc = cass_value_get_int16(lhs, reinterpret_cast<int16_t * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse int16 unsuccessful, column:" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            CassError rc = cass_value_get_int8(lhs, reinterpret_cast<int8_t * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse int16 unsuccessful, column:" + std::to_string(col));
            if (rc == CASS_ERROR_LIB_NULL_VALUE) return -1;
        }
        case CASS_VALUE_TYPE_LIST: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_MAP: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_SET: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_TUPLE: {
            TupleRow** ptr = (TupleRow**) data;
            auto last_elem = --metadata->at(col).pointer->end();
            if (last_elem == metadata->at(col).pointer->end()) throw ModuleException("Empty tuple");

            uint32_t internal_size = last_elem->position + last_elem->size;
            void *tuple_data = malloc(internal_size);

            *ptr = new TupleRow(metadata->at(col).pointer,internal_size, tuple_data);

            auto TFACT = TupleRowFactory(metadata->at(col).pointer);

            CassIterator* tuple_iterator = cass_iterator_from_tuple(lhs);
            if (!tuple_iterator) throw ModuleException("Cassandra to C: Data type is not tuple");
            /* Iterate over the tuple fields */
            uint32_t j = 0;
            while (cass_iterator_next(tuple_iterator)) {
                const CassValue* value = cass_iterator_get_value(tuple_iterator);
                char* pos_to_copy = (char*)tuple_data+metadata->at(col).pointer->at(j).position;
                TFACT.cass_to_c(value, pos_to_copy, j);
                ++j;
            }
            /* The tuple iterator needs to be freed */
            cass_iterator_free(tuple_iterator);
            return 0;
        }
        case CASS_VALUE_TYPE_UDT: {
            throw ModuleException("UDT not supported");
            //TODO REWRITE -> pass this to binary
        }
        case CASS_VALUE_TYPE_CUSTOM:
        case CASS_VALUE_TYPE_UNKNOWN:
        default:
            //TODO
            break;
    }
    return 0;
}



void TupleRowFactory::bind_tuple(CassStatement *statement, const TupleRow *row, u_int16_t offset, CassTuple * tuple) const {

    const std::vector<ColumnMeta> *localMeta = metadata.get();

    if(localMeta == nullptr) std::cout << "metadata is null" << std::endl;

    if (!localMeta)
        throw ModuleException("Tuple row, tuple_as_py: Null metadata");

    for (uint16_t n = 0; n < row->n_elem(); ++n) {

        const void *element_i = row->get_element(n);

        using namespace std;

        uint32_t bind_pos = n + offset;
        if (n >= localMeta->size())
            throw ModuleException(
                    "TupleRowFactory: Binding element on pos " + std::to_string(bind_pos) + " from a max " +
                    std::to_string(localMeta->size() + offset));
        if (element_i != nullptr) {
            switch (localMeta->at(n).type) {
                case CASS_VALUE_TYPE_VARCHAR:
                case CASS_VALUE_TYPE_TEXT:{
                    int64_t *addr = (int64_t *) row->get_element(n);
                    const char *d = reinterpret_cast<char *>(*addr);
                    cass_tuple_set_string(tuple, (size_t)n, d);
                    break;
                }
                case CASS_VALUE_TYPE_ASCII:
                case CASS_VALUE_TYPE_VARINT:
                case CASS_VALUE_TYPE_BIGINT: {
                    int64_t* value = (int64_t*) row->get_element(n);
                    cass_tuple_set_int64(tuple, (size_t)n, *value);
                    break;

                }
                case CASS_VALUE_TYPE_BLOB: {
                    int64_t * value = (int64_t*) row->get_element(n);
                    cass_tuple_set_int64(tuple, (size_t)n, *value);
                    break;
                }
                case CASS_VALUE_TYPE_BOOLEAN: {
                    cass_bool_t * value = (cass_bool_t*) row->get_element(n);
                    cass_tuple_set_int64(tuple, (size_t)n, *value);
                    break;
                }
                    //TODO parsed as uint32 or uint64 on different methods
                case CASS_VALUE_TYPE_COUNTER: {
                    int64_t * value = (int64_t*) row->get_element(n);
                    cass_tuple_set_int64(tuple, (size_t)n, *value);
                    break;
                }
                case CASS_VALUE_TYPE_DECIMAL: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_DOUBLE: {
                    double_t * value = (double_t*) row->get_element(n);
                    cass_tuple_set_double(tuple, (size_t)n, *value);
                    break;
                }
                case CASS_VALUE_TYPE_FLOAT: {
                    float_t * value = (float_t*) row->get_element(n);
                    cass_tuple_set_float(tuple, (size_t)n, *value);
                    break;
                }
                case CASS_VALUE_TYPE_INT: {
                    int32_t* value = (int32_t*) row->get_element(n);
                    cass_tuple_set_int32(tuple, (size_t)n, *value);
                    break;
                }
                case CASS_VALUE_TYPE_TIMESTAMP: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_UUID: {
                    const uint64_t **uuid = (const uint64_t **) row->get_element(n);

                    const uint64_t *time_and_version = *uuid;
                    const uint64_t *clock_seq_and_node = *uuid + 1;

                    CassUuid cass_uuid = {*time_and_version, *clock_seq_and_node};
                    cass_tuple_set_uuid(tuple, (size_t)n, cass_uuid);
                    break;
                }
                case CASS_VALUE_TYPE_TIMEUUID: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_INET: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_DATE: {

                    break;
                }
                case CASS_VALUE_TYPE_TIME: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_SMALL_INT: {
                    int16_t * value = (int16_t*) row->get_element(n);
                    cass_tuple_set_int16(tuple, (size_t)n, *value);
                    break;
                }
                case CASS_VALUE_TYPE_TINY_INT: {
                    int8_t * value = (int8_t*) row->get_element(n);
                    cass_tuple_set_int8(tuple, (size_t)n, *value);
                    break;
                }
                case CASS_VALUE_TYPE_LIST: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_MAP: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_SET: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_TUPLE: {
                    TupleRow **ptr = (TupleRow **) element_i;
                    const TupleRow *inner_data = *ptr;
                    auto TFACT = TupleRowFactory(metadata->at(n).pointer);
                    unsigned long n_types = metadata->at(n).pointer->size();
                    CassTuple *tuple = cass_tuple_new(n_types);
                    TFACT.bind_tuple(statement, inner_data, bind_pos, tuple);
                    cass_statement_bind_tuple(statement, bind_pos, tuple);
                    break;
                }
                case CASS_VALUE_TYPE_UDT: {
                    throw ModuleException("User defined types not supported");
                }
                case CASS_VALUE_TYPE_CUSTOM:
                case CASS_VALUE_TYPE_UNKNOWN:
                default:
                    //TODO
                    break;
            }
        }
        else {
            //Element is a nullptr
            CassError rc = cass_statement_bind_null(statement, bind_pos);
            CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [Null value], column:" +
                       localMeta->at(bind_pos).info[0]);
        }
        //else: key was nullptr and we don't bind anything
    }
}

void TupleRowFactory::bind(CassStatement *statement, const TupleRow *row, u_int16_t offset, CassTuple * tuple = NULL, std::string type = "NONE") const {

    const std::vector<ColumnMeta> *localMeta = metadata.get();

    if(localMeta == nullptr) std::cout << "metadata is null" << std::endl;

    if (!localMeta)
        throw ModuleException("Tuple row, tuple_as_py: Null metadata");

    for (uint16_t i = 0; i < row->n_elem(); ++i) {

        const void *element_i = row->get_element(i);

        using namespace std;

        uint32_t bind_pos = i + offset;
        if (i >= localMeta->size())
            throw ModuleException(
                    "TupleRowFactory: Binding element on pos " + std::to_string(bind_pos) + " from a max " +
                    std::to_string(localMeta->size() + offset));
        if (element_i != nullptr) {
            switch (localMeta->at(i).type) {
                case CASS_VALUE_TYPE_VARCHAR:
                case CASS_VALUE_TYPE_TEXT:{
                    int64_t *addr = (int64_t *) element_i;
                    const char *d = reinterpret_cast<char *>(*addr);
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_string(statement, bind_pos, d);
                        CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [text], column:" +
                                   localMeta->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_string(tuple, (size_t)i, d);
                    break;
                }
                case CASS_VALUE_TYPE_ASCII:
                case CASS_VALUE_TYPE_VARINT:
                case CASS_VALUE_TYPE_BIGINT: {
                    const int64_t *data = static_cast<const int64_t *>(element_i);
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_int64(statement, bind_pos,
                                                                 *data);//L means long long, K unsigned long long
                        CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [bigint/varint], column:" +
                                   localMeta->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_int64(tuple, (size_t)i, *data);
                    break;
                }
                case CASS_VALUE_TYPE_BLOB: {
                    unsigned char *byte_array;
                    byte_array = *(unsigned char **) element_i;
                    uint64_t *num_bytes = (uint64_t *) byte_array;
                    const unsigned char *bytes = byte_array + sizeof(uint64_t);
                    if(type == "NONE") {
                        cass_statement_bind_bytes(statement, bind_pos, bytes, *num_bytes);
                    }

                    else cass_tuple_set_bytes(tuple, (size_t)i, bytes, *num_bytes);
                    break;
                }
                case CASS_VALUE_TYPE_BOOLEAN: {
                    cass_bool_t b = cass_false;
                    const bool *bindbool = static_cast<const bool *>(element_i);

                    if (*bindbool) b = cass_true;
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_bool(statement, bind_pos, b);
                        CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [bool], column:" +
                                   localMeta->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_bool(tuple, (size_t)i, b);
                    break;
                }
                    //TODO parsed as uint32 or uint64 on different methods
                case CASS_VALUE_TYPE_COUNTER: {
                    const uint64_t *data = static_cast<const uint64_t *>(element_i);
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_int64(statement, bind_pos,
                                                                 *data);//L means long long, K unsigned long long
                        CHECK_CASS(
                                "TupleRowFactory: Cassandra binding query unsuccessful [counter as uint64], column:" +
                                localMeta->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_int64(tuple, (size_t)i, *data);
                    break;
                }
                case CASS_VALUE_TYPE_DECIMAL: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_DOUBLE: {
                    const double *data = static_cast<const double *>(element_i);
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_double(statement, bind_pos, *data);
                        CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [double], column:" +
                                   localMeta->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_double(tuple, (size_t)i, *data);
                    break;
                }
                case CASS_VALUE_TYPE_FLOAT: {
                    const float *data = static_cast<const float *>(element_i);
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_float(statement, bind_pos, *data);
                        CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [float], column:" +
                               localMeta->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_float(tuple, (size_t)i, *data);
                    break;
                }
                case CASS_VALUE_TYPE_INT: {
                    const int32_t *data = static_cast<const int32_t *>(element_i);
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_int32(statement, bind_pos, *data);
                        CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [int32], column:" +
                                   localMeta->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_int32(tuple, (size_t)i, *data);
                    break;
                }
                case CASS_VALUE_TYPE_TIMESTAMP: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_UUID: {
                    const uint64_t **uuid = (const uint64_t **) element_i;

                    const uint64_t *time_and_version = *uuid;
                    const uint64_t *clock_seq_and_node = *uuid + 1;

                    CassUuid cass_uuid = {*time_and_version, *clock_seq_and_node};
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_uuid(statement, bind_pos, cass_uuid);
                        CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [UUID], column:" +
                               metadata->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_uuid(tuple, (size_t)i, cass_uuid);
                    break;
                }
                case CASS_VALUE_TYPE_TIMEUUID: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_INET: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_DATE: {

                    break;
                }
                case CASS_VALUE_TYPE_TIME: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_SMALL_INT: {
                    const int16_t *data = static_cast<const int16_t *>(element_i);
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_int16(statement, bind_pos, *data);
                        CHECK_CASS(
                                "TupleRowFactory: Cassandra binding query unsuccessful [small int as int16], column:" +
                                localMeta->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_int16(tuple, (size_t)i, *data);
                    break;
                }
                case CASS_VALUE_TYPE_TINY_INT: {

                    const int8_t *data = static_cast<const int8_t *>(element_i);
                    if(type == "NONE") {
                        CassError rc = cass_statement_bind_int8(statement, bind_pos, *data);
                        CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [tiny int as int8], column:" +
                                   localMeta->at(bind_pos).info[0]);
                    }
                    else cass_tuple_set_int8(tuple, (size_t)i, *data);
                    break;
                }
                case CASS_VALUE_TYPE_LIST: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_MAP: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_SET: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_TUPLE: {
                    TupleRow** ptr = (TupleRow**) element_i;
                    const TupleRow* inner_data = *ptr;
                    auto TFACT = TupleRowFactory(metadata->at(i).pointer);
                    unsigned long n_types = metadata->at(i).pointer->size();
                    CassTuple* tuple = cass_tuple_new(n_types);
                    TFACT.bind(statement,inner_data,bind_pos, tuple, "TUPLE");
                    type = "NONE";
                    cass_statement_bind_tuple(statement, bind_pos, tuple);
                    break;
                }
                case CASS_VALUE_TYPE_UDT: {
                    throw ModuleException("User defined types not supported");
                }
                case CASS_VALUE_TYPE_CUSTOM:
                case CASS_VALUE_TYPE_UNKNOWN:
                default:
                    //TODO
                    break;
            }
        } else {
            //Element is a nullptr
            CassError rc = cass_statement_bind_null(statement, bind_pos);
            CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [Null value], column:" +
                       localMeta->at(bind_pos).info[0]);
        }
        //else: key was nullptr and we don't bind anything
    }
}
