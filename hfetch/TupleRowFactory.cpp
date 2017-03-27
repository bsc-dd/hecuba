#include "TupleRowFactory.h"

/***
 * Builds a tuple factory to retrieve tuples based on rows and keys
 * extracting the information from Cassandra to decide the types to be used
 * @param table_meta Holds the table information
 */
TupleRowFactory::TupleRowFactory(std::shared_ptr<const std::vector<ColumnMeta> > row_info) {

    if (row_info->empty()) {
        throw ModuleException("Tuple factory: Table metadata empty");
    }
    this->metadata= row_info;
    this->total_bytes = 0;
    ColumnMeta last_meta = row_info->at(row_info->size()-1); //*(--this->metadata.get()->end()); TODO
    total_bytes=last_meta.position+last_meta.size;
}

/***
 * Returns the allocation number of bytes required to allocate a type of data
 * @param VT Cassandra Type
 * @return Allocation size needed to store data of type VT
 */
uint16_t TupleRowFactory::compute_size_of(const CassValueType VT) const {
    switch (VT) {
        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_ASCII: {
            return sizeof(char *);
        }
        case CASS_VALUE_TYPE_VARINT:
        case CASS_VALUE_TYPE_BIGINT: {
            return sizeof(int64_t);
        }
        case CASS_VALUE_TYPE_BLOB: {
            return sizeof(unsigned char *);
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
            return sizeof(bool);
        }
        case CASS_VALUE_TYPE_COUNTER: {
            return sizeof(uint32_t);
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //TODO
            std::cerr << "Parse decimals data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            return sizeof(double);
        }
        case CASS_VALUE_TYPE_FLOAT: {
            return sizeof(float);
        }
        case CASS_VALUE_TYPE_INT: {
            return sizeof(int32_t);
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {
            //TODO
            std::cerr << "Timestamp data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_UUID: {
            //TODO

            std::cerr << "UUID data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_TIMEUUID: {
            std::cerr << "TIMEUUID data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_INET: {

            //TODO
            std::cerr << "INET data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_DATE: {
            std::cerr << "Date data type supported yet" << std::endl;

            //TODO
            break;
        }
        case CASS_VALUE_TYPE_TIME: {
            std::cerr << "Time data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_SMALL_INT: {
            return sizeof(int16_t);
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            return sizeof(int8_t);
        }
        case CASS_VALUE_TYPE_LIST: {
            std::cerr << "List data type supported yet" << std::endl;
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_MAP: {
            std::cerr << "Map data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_SET: {
            std::cerr << "Set data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_TUPLE: {
            //TODO

            std::cerr << "Tuple data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_UDT:
        case CASS_VALUE_TYPE_CUSTOM:
        case CASS_VALUE_TYPE_UNKNOWN:
        default:
            throw ModuleException("Unknown data type, can't parse");
            //TODO
    }
    return 0;
}



/*** TUPLE BUILDERS ***/

/***
 * Build a tuple taking the given buffer as its internal representation
 * @param data Valid pointer to the values
 * @return TupleRow with the buffer as its inner data
 * @post The TupleRow now owns the data and this cannot be freed
 */
TupleRow *TupleRowFactory::make_tuple(void * data) {
    return  new TupleRow(metadata, total_bytes, data);
}


/***
 * Build a tuple from the given Cassandra result row using the factory's metadata
 * @param row Contains the same number of columns than the metadata
 * @return TupleRow with a copy of the values inside the row
 * @post The row can be freed
 */
TupleRow *TupleRowFactory::make_tuple(const CassRow *row) {
    if (!row) return  NULL;
    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, make tuple from CassRow: Null metadata");

    char *buffer = (char *) malloc(total_bytes);
    uint16_t i = 0;
    CassIterator *it = cass_iterator_from_row(row);
    while (cass_iterator_next(it)) {
        if (i>=localMeta->size())
            throw ModuleException("TupleRowFactory: Make tuple from CassRow: Access metadata at "+std::to_string(i)+" from a max "+std::to_string(localMeta->size()));
        cass_to_c(cass_iterator_get_column(it), buffer + localMeta->at(i).position, i);
        if (localMeta->at(i).position >= total_bytes)
            throw ModuleException("TupleRowFactory: Make tuple from CassRow: Writing on byte "+std::to_string(localMeta->at(i).position)+" from a total of "+std::to_string(total_bytes));
        if (i > localMeta->size())
            throw ModuleException("TupleRowFactory: Query has more columns than the ones retrieved from Cassandra");
        ++i;
    }
    cass_iterator_free(it);
    return new TupleRow(metadata, total_bytes, buffer);
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
    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, cass_to_c: Null metadata");
    if (col < 0 || col >= (int32_t) localMeta->size()) {
        throw ModuleException("TupleRowFactory: Cass to C: Asked for column "+std::to_string(col)+" but only "+std::to_string(localMeta->size())+" are present");
    }

    if (cass_value_is_null(lhs)) {
        memset(data, 0, compute_size_of(localMeta->at(col).type));
        return 0;
    }
    switch (localMeta->at(col).type) {
        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_ASCII: {
            const char *l_temp;
            size_t l_size;
            CassError rc = cass_value_get_string(lhs,&l_temp , &l_size);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse text unsuccessful, column"+std::to_string(col));
            char *permanent = (char*) malloc(l_size+1);
            memcpy(permanent, l_temp,l_size);
            permanent[l_size] = '\0';
            memcpy(data,&permanent,sizeof(char*));
            return 0;
        }
        case CASS_VALUE_TYPE_VARINT:
        case CASS_VALUE_TYPE_BIGINT: {
            int64_t *p = static_cast<int64_t * >(data);
            CassError rc = cass_value_get_int64(lhs, p);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse bigint/varint unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_BLOB: {
            const unsigned char *l_temp;
            size_t l_size;
            CassError rc = cass_value_get_bytes(lhs, &l_temp, &l_size);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse bytes unsuccessful, column:"+std::to_string(col));

            //Allolcate space for the bytes
            char *permanent = (char*) malloc(l_size+sizeof(uint32_t));
            //TODO make sure l_size < uint32 max
            uint32_t int_size = (uint32_t) l_size;

            //copy num bytes
            memcpy(permanent,&int_size,sizeof(uint32_t));

            //copy bytes
            memcpy(permanent+sizeof(uint32_t), l_temp,l_size);

            //copy pointer to payload
            memcpy(data,&permanent,sizeof(char*));
            return 0;
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
            cass_bool_t b;
            CassError rc = cass_value_get_bool(lhs, &b);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse bool unsuccessful, column:"+std::to_string(col));
            bool *p = static_cast<bool *>(data);
            if (b == cass_true) *p = true;
            else *p = false;
            return 0;
        }
        case CASS_VALUE_TYPE_COUNTER: {
            CassError rc = cass_value_get_uint32(lhs, reinterpret_cast<uint32_t *>(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse counter as uint32 unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //TODO
            //decimal.Decimal
            break;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            CassError rc = cass_value_get_double(lhs, reinterpret_cast<double *>(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse double unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_FLOAT: {
            CassError rc = cass_value_get_float(lhs, reinterpret_cast<float * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse float unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_INT: {
            CassError rc = cass_value_get_int32(lhs, reinterpret_cast<int32_t * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse int32 unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_UUID: {
            //TODO
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
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_TIME: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_SMALL_INT: {
            CassError rc = cass_value_get_int16(lhs, reinterpret_cast<int16_t * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse int16 unsuccessful, column:"+std::to_string(col));
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            CassError rc = cass_value_get_int8(lhs, reinterpret_cast<int8_t * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse int16 unsuccessful, column:"+std::to_string(col));
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
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_UDT:
        case CASS_VALUE_TYPE_CUSTOM:
        case CASS_VALUE_TYPE_UNKNOWN:
        default:
            break;
    }
    return 0;
}


/***
 * Binds the tuple row data to the statement, using the offset to compute the position inside the query where the element needs to be bind
 * @param statement contains the query with elements to be binded
 * @param row data to use to bind in the statement
 * @param offset starting position inside the query from which the elements need to be bind
 */
void TupleRowFactory::bind( CassStatement *statement,const TupleRow *row,  u_int16_t offset) const  {

    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, tuple_as_py: Null metadata");

    for (uint16_t i = 0; i < row->n_elem(); ++i) {
        const void *key = row->get_element(i);

        uint32_t bind_pos = i + offset;
        if (i>=localMeta->size())
            throw ModuleException("TupleRowFactory: Binding element on pos "+std::to_string(bind_pos)+" from a max "+std::to_string(localMeta->size()));

        switch (localMeta->at(i).type) {
            case CASS_VALUE_TYPE_VARCHAR:
            case CASS_VALUE_TYPE_TEXT:
            case CASS_VALUE_TYPE_ASCII: {
                int64_t *addr = (int64_t *) key;
                const char *d = reinterpret_cast<char *>(*addr);
                CassError rc = cass_statement_bind_string(statement, bind_pos, d);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [text], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_VARINT:
            case CASS_VALUE_TYPE_BIGINT: {
                const int64_t *data = static_cast<const int64_t *>(key);
                CassError rc = cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [bigint/varint], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_BLOB: {
                int64_t  *addr = (int64_t*) key;
                const unsigned char *d = reinterpret_cast<char unsigned *>(*addr);
                uint32_t nbyes = *reinterpret_cast<uint32_t *>(*addr);
                CassError rc = cass_statement_bind_bytes(statement,bind_pos,d+sizeof(uint32_t),nbyes);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [bytes], column:"+localMeta->at(bind_pos).name);


                break;
            }
            case CASS_VALUE_TYPE_BOOLEAN: {
                cass_bool_t b = cass_false;
                const bool *bindbool = static_cast<const bool *>(key);
                if (*bindbool) b = cass_true;
                CassError rc = cass_statement_bind_bool(statement, bind_pos, b);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [bool], column:"+localMeta->at(bind_pos).name);
                break;
            }
            //TODO parsed as uint32 or uint64 on different methods
            case CASS_VALUE_TYPE_COUNTER: {
                const uint64_t *data = static_cast<const uint64_t *>(key);
                CassError rc = cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [counter as uint64], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_DECIMAL: {
                //decimal.Decimal
                //TODO
                break;
            }
            case CASS_VALUE_TYPE_DOUBLE: {
                const double *data = static_cast<const double *>(key);
                CassError rc = cass_statement_bind_double(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [double], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_FLOAT: {
                const float *data = static_cast<const float *>(key);
                CassError rc = cass_statement_bind_float(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [float], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_INT: {
                const int32_t *data = static_cast<const int32_t *>(key);
                CassError rc = cass_statement_bind_int32(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [int32], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_TIMESTAMP: {
                //TODO
                break;
            }
            case CASS_VALUE_TYPE_UUID: {
                //TODO
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
                const int16_t *data = static_cast<const int16_t *>(key);
                CassError rc = cass_statement_bind_int16(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [small int as int16], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_TINY_INT: {
                const int8_t *data = static_cast<const int8_t *>(key);
                CassError rc = cass_statement_bind_int8(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [tiny int as int8], column:"+localMeta->at(bind_pos).name);
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
                //TODO
                break;
            }
            case CASS_VALUE_TYPE_UDT:
            case CASS_VALUE_TYPE_CUSTOM:
            case CASS_VALUE_TYPE_UNKNOWN:
            default:
                //TODO
                break;
        }
    }
}
