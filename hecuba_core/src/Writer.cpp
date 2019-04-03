#include "Writer.h"

#define default_writer_buff 1000
#define default_writer_callbacks 16


Writer::Writer(const TableMetadata *table_meta, CassSession *session,
               std::map<std::string, std::string> &config) {

    int32_t buff_size = default_writer_buff;
    int32_t max_callbacks = default_writer_callbacks;


    if (config.find("writer_par") != config.end()) {
        std::string max_callbacks_str = config["writer_par"];
        try {
            max_callbacks = std::stoi(max_callbacks_str);
            if (max_callbacks <= 0) throw ModuleException("Writer parallelism value must be > 0");
        }
        catch (std::exception &e) {
            std::string msg(e.what());
            msg += " Malformed value in config for writer_par";
            throw ModuleException(msg);
        }
    }

    if (config.find("writer_buffer") != config.end()) {
        std::string buff_size_str = config["writer_buffer"];
        try {
            buff_size = std::stoi(buff_size_str);
            if (buff_size < 0) throw ModuleException("Writer buffer value must be >= 0");
        }
        catch (std::exception &e) {
            std::string msg(e.what());
            msg += " Malformed value in config for writer_buffer";
            throw ModuleException(msg);
        }
    }

    this->session = session;
    this->table_metadata = table_meta;
    this->k_factory = new TupleRowFactory(table_meta->get_keys());
    this->v_factory = new TupleRowFactory(table_meta->get_values());

    CassFuture *future = cass_session_prepare(session, table_meta->get_insert_query());
    CassError rc = cass_future_error_code(future);
    CHECK_CASS("writer cannot prepare: ");
    this->prepared_query = cass_future_get_prepared(future);
    cass_future_free(future);
    this->data.set_capacity(buff_size);
    this->max_calls = (uint32_t) max_callbacks;
    this->ncallbacks = 0;
    this->error_count = 0;
}


Writer::~Writer() {
    flush_elements();
    if (this->prepared_query != NULL) {
        cass_prepared_free(this->prepared_query);
        prepared_query = NULL;
    }
    delete (this->k_factory);
    delete (this->v_factory);
}


void Writer::flush_elements() {
    while (!data.empty() || ncallbacks > 0) {
        if (ncallbacks < max_calls) {
            ncallbacks++;
            call_async();
        } else std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


void Writer::callback(CassFuture *future, void *ptr) {
    void **data = reinterpret_cast<void **>(ptr);
    assert(data != NULL && data[0] != NULL);
    Writer *W = (Writer *) data[0];

    CassError rc = cass_future_error_code(future);
    if (rc != CASS_OK) {
        std::string message(cass_error_desc(rc));
        const char *dmsg;
        size_t l;
        cass_future_error_message(future, &dmsg, &l);
        std::string msg2(dmsg, l);
        W->set_error_occurred("Writer callback: " + message + "  " + msg2, data[1], data[2]);
    } else {
        delete ((TupleRow *) data[1]);
        delete ((TupleRow *) data[2]);
        W->call_async();
    }
    free(data);

}


void Writer::set_error_occurred(std::string error, const void *keys_p, const void *values_p) {
    ++error_count;

    if (error_count > MAX_ERRORS) {
        --ncallbacks;
        throw ModuleException("Try # " + std::to_string(MAX_ERRORS) + " :" + error);
    } else {
        std::cerr << "Connectivity problems: " << error_count << " " << error << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    const TupleRow *keys = (TupleRow *) keys_p;
    const TupleRow *values = (TupleRow *) values_p;

    /** write the data which hasn't been written successfully **/
    CassStatement *statement = cass_prepared_bind(prepared_query);


    this->k_factory->bind(statement, keys, 0, NULL, "NONE");
    this->v_factory->bind(statement, values, this->k_factory->n_elements(), NULL, "NONE");


    CassFuture *query_future = cass_session_execute(session, statement);

    cass_statement_free(statement);

    const void **data = (const void **) malloc(sizeof(void *) * 3);
    data[0] = this;
    data[1] = keys_p;
    data[2] = values_p;
    cass_future_set_callback(query_future, callback, data);
    cass_future_free(query_future);
}


void Writer::write_to_cassandra(const TupleRow *keys, const TupleRow *values) {
    std::pair<const TupleRow *, const TupleRow *> item = std::make_pair(new TupleRow(keys), new TupleRow(values));
    data.push(item);
    if (ncallbacks < max_calls) {
        ncallbacks++;
        call_async();
    }
}

void Writer::write_to_cassandra(void *keys, void *values) {
    const TupleRow *k = k_factory->make_tuple(keys);
    const TupleRow *v = v_factory->make_tuple(values);
    this->write_to_cassandra(k, v);
    delete (k);
    delete (v);
}

void Writer::call_async() {

    //current write data
    std::pair<const TupleRow *, const TupleRow *> item;
    if (!data.try_pop(item)) {
        ncallbacks--;
        return;
    }

    CassStatement *statement = cass_prepared_bind(prepared_query);

    this->k_factory->bind(statement, item.first, 0, NULL, "NONE");
    this->v_factory->bind(statement, item.second, this->k_factory->n_elements(), NULL, "NONE");


    CassFuture *query_future = cass_session_execute(session, statement);
    cass_statement_free(statement);

    const void **data = (const void **) malloc(sizeof(void *) * 3);
    data[0] = this;
    data[1] = item.first;
    data[2] = item.second;

    cass_future_set_callback(query_future, callback, data);
    cass_future_free(query_future);
}

