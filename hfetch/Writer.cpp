#include "Writer.h"

Writer::Writer(uint16_t buff_size, uint16_t max_callbacks, TupleRowFactory &key_factory, TupleRowFactory &value_factory,
               CassSession *session,
               std::string query) {
    this->session = session;
    this->k_factory = key_factory;
    this->v_factory = value_factory;
    CassFuture *future = cass_session_prepare(session, query.c_str());
    CassError rc = cass_future_error_code(future);
    CHECK_CASS("writer cannot prepare");
    this->prepared_query = cass_future_get_prepared(future);
    cass_future_free(future);
    this->data.set_capacity(buff_size);
    this->max_calls = max_callbacks;
    this->ncallbacks=0;
}


Writer::~Writer() {
    flush_elements();
    if (this->prepared_query!=NULL) cass_prepared_free(this->prepared_query);
}


void Writer::flush_elements() {
    while (!data.empty()) {
        if (ncallbacks<max_calls) {
            ncallbacks++;
            call_async();
        }
    }

   while (ncallbacks>0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
}


static void callback(CassFuture *future, void *ptr) {
    CassError rc = cass_future_error_code(future);
    if (rc != CASS_OK) {
        std::string message(cass_error_desc(rc));
        const char *dmsg;
        size_t l;
        cass_future_error_message(future, &dmsg, &l);
        std::string msg2(dmsg,l);
        throw ModuleException("Writer callback: " + message +"  "+msg2);
    }

    Writer *W = (Writer*) ptr;
    W->call_async();
}


void Writer::write_to_cassandra(const TupleRow *keys,const TupleRow *values) {
    if (ncallbacks < max_calls) {
        ncallbacks++;
        auto item = std::make_pair(keys,values);
        data.push(item);
        call_async();
    } else {
        auto item = std::make_pair(keys,values);
        data.push(item);
    }
}


void Writer::call_async() {
    std::pair<const TupleRow*,const TupleRow*> item;
    if (!data.try_pop(item)) {
        ncallbacks--;
        return;
    }
    CassStatement *statement = cass_prepared_bind(prepared_query);

    bind(statement, item.first, k_factory, 0);
    bind(statement, item.second, v_factory, k_factory.n_elements());

    CassFuture *query_future = cass_session_execute(session, statement);

    cass_statement_free(statement);

    cass_future_set_callback(query_future,callback, this);
    cass_future_free(query_future);
}



//Same as CacheTable.cpp
void Writer::bind(CassStatement *statement, const TupleRow *tuple_row, const TupleRowFactory &factory, uint16_t offset) {
    for (uint16_t i = 0; i < tuple_row->n_elem(); ++i) {

        const void *key = tuple_row->get_element(i);
        uint16_t bind_pos = i + offset;
        switch (factory.get_type(i)) {
            case CASS_VALUE_TYPE_VARCHAR:
            case CASS_VALUE_TYPE_TEXT:
            case CASS_VALUE_TYPE_ASCII: {
                int64_t  *addr = (int64_t*) key;
                const char *d = reinterpret_cast<char *>(*addr);
                cass_statement_bind_string(statement, bind_pos, d);
                break;
            }
            case CASS_VALUE_TYPE_VARINT:
            case CASS_VALUE_TYPE_BIGINT: {
                const int64_t *data = static_cast<const int64_t *>(key);
                cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                break;
            }
            case CASS_VALUE_TYPE_BLOB: {
                //cass_statement_bind_bytes(statement,bind_pos,key,n_elem);
                break;
            }
            case CASS_VALUE_TYPE_BOOLEAN: {
                cass_bool_t b = cass_false;
                const bool *bindbool = static_cast<const bool *>(key);
                if (*bindbool) b = cass_true;
                cass_statement_bind_bool(statement, bind_pos, b);
                break;
            }
            case CASS_VALUE_TYPE_COUNTER: {
                const uint64_t *data = static_cast<const uint64_t *>(key);
                cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                break;
            }
            case CASS_VALUE_TYPE_DECIMAL: {
                //decimal.Decimal
                break;
            }
            case CASS_VALUE_TYPE_DOUBLE: {
                const double *data = static_cast<const double *>(key);
                cass_statement_bind_double(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_FLOAT: {
                const float *data = static_cast<const float *>(key);
                cass_statement_bind_float(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_INT: {
                const int32_t *data = static_cast<const int32_t *>(key);
                cass_statement_bind_int32(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_TIMESTAMP: {

                break;
            }
            case CASS_VALUE_TYPE_UUID: {

                break;
            }
            case CASS_VALUE_TYPE_TIMEUUID: {

                break;
            }
            case CASS_VALUE_TYPE_INET: {

                break;
            }
            case CASS_VALUE_TYPE_DATE: {

                break;
            }
            case CASS_VALUE_TYPE_TIME: {

                break;
            }
            case CASS_VALUE_TYPE_SMALL_INT: {
                const int16_t *data = static_cast<const int16_t *>(key);
                cass_statement_bind_int16(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_TINY_INT: {
                const int8_t *data = static_cast<const int8_t *>(key);
                cass_statement_bind_int8(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_LIST: {
                break;
            }
            case CASS_VALUE_TYPE_MAP: {

                break;
            }
            case CASS_VALUE_TYPE_SET: {

                break;
            }
            case CASS_VALUE_TYPE_TUPLE: {
                break;
            }
            default://CASS_VALUE_TYPE_UDT|CASS_VALUE_TYPE_CUSTOM|CASS_VALUE_TYPE_UNKNOWN:
                break;
        }
    }
}