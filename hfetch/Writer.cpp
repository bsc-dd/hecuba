#include "Writer.h"

Writer::Writer(uint16_t buff_size, uint16_t max_callbacks, const TupleRowFactory &key_factory,
               const TupleRowFactory &value_factory,
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
    this->ncallbacks = 0;
}


Writer::~Writer() {
    flush_elements();
    if (this->prepared_query != NULL) cass_prepared_free(this->prepared_query);
}


void Writer::flush_elements() {
    while (!data.empty()) {
        if (ncallbacks < max_calls) {
            ncallbacks++;
            call_async();
        }
    }

    while (ncallbacks > 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
}


static void callback(CassFuture *future, void *ptr) {
    CassError rc = cass_future_error_code(future);
    if (rc != CASS_OK) {
        std::string message(cass_error_desc(rc));
        const char *dmsg;
        size_t l;
        cass_future_error_message(future, &dmsg, &l);
        std::string msg2(dmsg, l);
        throw ModuleException("Writer callback: " + message + "  " + msg2);
    }

    Writer *W = (Writer *) ptr;
    W->call_async();
}


void Writer::write_to_cassandra(const TupleRow *keys, const TupleRow *values) {
    if (ncallbacks < max_calls) {
        ncallbacks++;
        auto item = std::make_pair(keys, values);
        data.push(item);
        call_async();
    } else {
        auto item = std::make_pair(keys, values);
        data.push(item);
    }
}


void Writer::call_async() {
    std::pair<const TupleRow *, const TupleRow *> item;
    if (!data.try_pop(item)) {
        ncallbacks--;
        return;
    }
    CassStatement *statement = cass_prepared_bind(prepared_query);

    this->k_factory.bind(statement,item.first,0);
    this->v_factory.bind(statement,item.second,this->k_factory.n_elements());


    CassFuture *query_future = cass_session_execute(session, statement);

    cass_statement_free(statement);

    cass_future_set_callback(query_future, callback, this);
    cass_future_free(query_future);
}

