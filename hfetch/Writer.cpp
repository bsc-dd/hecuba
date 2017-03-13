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

    Writer::currentUnit *data = (Writer::currentUnit *)ptr;


    Writer *W = data->W;

    CassError rc = cass_future_error_code(future);
    if (rc != CASS_OK) {
        std::string message(cass_error_desc(rc));
        const char *dmsg;
        size_t l;
        cass_future_error_message(future, &dmsg, &l);
        std::string msg2(dmsg, l);
        W->set_error("Writer callback: " + message + "  " + msg2,data->key,data->value);
    }
    W->call_async();
}


void Writer::set_error(std::string error,const TupleRow* keys, const TupleRow* values) {
    ++error_count;
    if (error_count>MAX_ERRORS) {
        --ncallbacks;
        throw ModuleException("Try # "+std::to_string(MAX_ERRORS)+" :"+error);
    }
    else {
        std::cout << "Connectivity problems: " << error << std::endl;
        sleep(1);
    }
    std::pair<const TupleRow *, const TupleRow *> item = std::make_pair(keys, values);
    data.push(item);
    if (ncallbacks==1) call_async();
    else --ncallbacks;
}

void Writer::write_to_cassandra(const TupleRow *keys, const TupleRow *values) {
    std::pair<const TupleRow *, const TupleRow *> item = std::make_pair(keys, values);
    data.push(item);
    if (ncallbacks < max_calls) {
        ncallbacks++;
        call_async();
    }
}


void Writer::call_async() {


    //current write data
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

 //   std::pair<Writer*,std::pair<const TupleRow *, const TupleRow *> >data(this,item);
    currentUnit *t = new currentUnit(this,item.first,item.second);

    cass_future_set_callback(query_future, callback,(void*) t);
    cass_future_free(query_future);
}

