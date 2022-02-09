#include "Writer.h"

#define default_writer_buff 1000
#define default_writer_callbacks 16


/* Thread to process the pending data to be sent to cassandra */
void Writer::async_query_thread_code() {
    while(!finish_async_query_thread) {
        while((ncallbacks < max_calls) && !data.empty()) {
            call_async();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

Writer::Writer(const TableMetadata *table_meta, CassSession *session,
               std::map<std::string, std::string> &config) {


    //std::cout<< " WRITER: Constructor "<< std::endl;
    int32_t buff_size = default_writer_buff;
    int32_t max_callbacks = default_writer_callbacks;
    this->disable_timestamps = false;

    if (config.find("timestamped_writes") != config.end()) {
        std::string check_timestamps = config["timestamped_writes"];
        std::transform(check_timestamps.begin(), check_timestamps.end(), check_timestamps.begin(), ::tolower);
        if (check_timestamps == "false" || check_timestamps == "no")
            disable_timestamps = true;
    }

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
    this->timestamp_gen = new TimestampGenerator();
    this->lazy_write_enabled = false; // Disabled by default, will be enabled on ArrayDataStore
    this->dirty_blocks = new tbb::concurrent_hash_map <const TupleRow *, const TupleRow *, Writer::HashCompare >();
    this->finish_async_query_thread = false;
    this->async_query_thread_created = false;
}


Writer::~Writer() {
    //std::cout<< " WRITER: Destructor "<< std::endl;
    if (this->async_query_thread_created){
        wait_writes_completion(); // WARNING! It is necessary to wait for ALL CALLBACKS to finish, because the 'data' structure required by the callback will dissapear with this destructor
        auto async_query_thread_id = this->async_query_thread.get_id();
        this->finish_async_query_thread = true;
        this->async_query_thread.join();
        //std::cout<< " WRITER: Finished thread "<< async_query_thread_id << std::endl;
    }
    if (this->prepared_query != NULL) {
        cass_prepared_free(this->prepared_query);
        prepared_query = NULL;
    }
    delete (this->k_factory);
    delete (this->v_factory);
    delete (this->timestamp_gen);
    delete (this->dirty_blocks);
}


void Writer::set_timestamp_gen(TimestampGenerator *time_gen) {
    delete(this->timestamp_gen);
    this->timestamp_gen = time_gen;
}

/* Queue a new pair {keys, values} into the 'data' queue to be executed later.
 * Args are copied, therefore they may be deleted after calling this method. */
void Writer::queue_async_query( const TupleRow *keys, const TupleRow *values) {
    TupleRow *queued_keys = new TupleRow(keys);
    if (!disable_timestamps) queued_keys->set_timestamp(timestamp_gen->next()); // Set write time
    std::pair<const TupleRow *, const TupleRow *> item = std::make_pair(queued_keys, new TupleRow(values));

    //std::cout<< "  Writer::flushing item created pair"<<std::endl;
    data.push(item);
}

void Writer::flush_dirty_blocks() {
    //std::cout<< "Writer::flush_dirty_blocks "<<std::endl;
    int n = 0;
    for (auto x : *dirty_blocks) {
        //std::cout<< "  Writer::flushing item "<<std::endl;
        n ++;
        queue_async_query(x.first, x.second);
        delete(x.first);
        delete(x.second);
    }
    dirty_blocks->clear();
    //std::cout<< "Writer::flush_dirty_blocks "<< n << " blocks FLUSHED"<<std::endl;
}

// flush all the pending write requests: send them to Cassandra driver and wait for finalization (called from outside)
void Writer::flush_elements() {
    wait_writes_completion();
}

// wait for callbacks execution for all sent write requests
void Writer::wait_writes_completion(void) {
    flush_dirty_blocks();
    //std::cout<< "Writer::wait_writes_completion * Waiting for "<< data.size() << " Pending "<<ncallbacks<<" callbacks" <<" inflight"<<std::endl;
    while(!data.empty() || ncallbacks>0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    //std::cout<< "Writer::wait_writes_completion2* Waiting for "<< data.size() << " Pending "<<ncallbacks<<" callbacks" <<" inflight"<<std::endl;
}

void Writer::callback(CassFuture *future, void *ptr) {
    void **data = reinterpret_cast<void **>(ptr);
    assert(data != NULL && data[0] != NULL);
    Writer *W = (Writer *) data[0];

    //std::cout<< "Writer::callback"<< std::endl;
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
        W->ncallbacks--;
    }
    free(data);
}


void Writer::async_query_execute(const TupleRow *keys, const TupleRow *values) {

    CassStatement *statement;
    // Check if it is writing the whole set of values or just a single one
    if (table_metadata->get_values()->size() > values->n_elem()) { // Single value written
        if (values->n_elem() > 1)
            throw ModuleException("async_query_execute: only supports 1 or all attributes write");

        const CassPrepared *prepared_query;
        ColumnMeta cm = values->get_metadata_element(0);
        const char* insert_q = table_metadata->get_partial_insert_query(cm.info["name"]);
        CassFuture *future = nullptr;
        try {
            future = cass_session_prepare(session, insert_q);
        } catch (std::exception &e) {
            std::string msg(e.what());
            msg += " Problem in execute " + std::string(insert_q);
            throw ModuleException(msg);
        }
        CassError rc = cass_future_error_code(future);
        CHECK_CASS("writer cannot prepare: ");
        prepared_query = cass_future_get_prepared(future);
        statement = cass_prepared_bind(prepared_query);
        this->k_factory->bind(statement, keys, 0); //error
        TupleRowFactory * v_single_factory = new TupleRowFactory(table_metadata->get_single_value(cm.info["name"].c_str()));
        v_single_factory->bind(statement, values, this->k_factory->n_elements());
        delete(v_single_factory);

    } else { // Whole row written
        statement = cass_prepared_bind(prepared_query);
        this->k_factory->bind(statement, keys, 0); //error
        this->v_factory->bind(statement, values, this->k_factory->n_elements());
    }

    if (!this->disable_timestamps) {
        cass_statement_set_timestamp(statement, keys->get_timestamp());
    }

    CassFuture *query_future = cass_session_execute(session, statement);
    cass_statement_free(statement);

    const void **data = (const void **) malloc(sizeof(void *) * 3);
    data[0] = this;
    data[1] = keys;
    data[2] = values;

    cass_future_set_callback(query_future, callback, data);
    cass_future_free(query_future);
}

void Writer::set_error_occurred(std::string error, const void *keys_p, const void *values_p) {
    ++error_count;

    if (error_count > MAX_ERRORS) {
        --ncallbacks;
        throw ModuleException("Try # " + std::to_string(MAX_ERRORS) + " :" + error);
    } else {
        std::cerr << "Connectivity problems: " << error_count << " " << error << std::endl;
        std::cerr << "  WARNING: We can NOT ensure write requests order->POTENTIAL INCONSISTENCY"<<std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    const TupleRow *keys = (TupleRow *) keys_p;
    const TupleRow *values = (TupleRow *) values_p;

    /** write the data which hasn't been written successfully **/
    async_query_execute(keys, values);
}

void Writer::enable_lazy_write(void) {
    this->lazy_write_enabled = true;
}

void Writer::disable_lazy_write(void) {
    if (this->lazy_write_enabled) {
        flush_dirty_blocks();
        this->lazy_write_enabled = false;
    }
}

void Writer::write_to_cassandra(const TupleRow *keys, const TupleRow *values) {


    this->async_query_thread_lock.lock();
    //std::cout<< " WRITER: write_to_cassandra" << std::endl;
    if (this->async_query_thread_created == false) {
        this->async_query_thread_created = true;
        this->async_query_thread_lock.unlock();
        this->async_query_thread = std::thread(&Writer::async_query_thread_code, this);
        //std::cout<< " WRITER: Created thread "<< this->async_query_thread.get_id() << std::endl;
    } else {
        this->async_query_thread_lock.unlock();
    }
    if (lazy_write_enabled) {
        //put into dirty_blocks. Skip the repeated 'keys' requests replacing the value.
        tbb::concurrent_hash_map <const TupleRow*, const TupleRow*, Writer::HashCompare>::accessor a;

        if (!dirty_blocks->find(a, keys)) {
            const TupleRow* k = new TupleRow(keys);
            const TupleRow* v = new TupleRow(values);
            if (dirty_blocks->insert(a, k)) {
                a->second = v;
            }
        } else { // Replace value
            delete a->second;
            const TupleRow* v = new TupleRow(values);
            a->second = v;
        }

        if (dirty_blocks->size() > max_calls) {//if too many dirty_blocks
            flush_dirty_blocks();
        }
    } else {
        queue_async_query(keys, values);
    }
}

void Writer::write_to_cassandra(void *keys, void *values) {
    const TupleRow *k = k_factory->make_tuple(keys);
    const TupleRow *v = v_factory->make_tuple(values);
    this->write_to_cassandra(k, v);
    delete (k);
    delete (v);
}

void Writer::write_to_cassandra(void *keys, void *values , const char *value_name) {
    // When trying to write a single attribute of the cassandra table we MUST
    // DISABLE the dirty cache as the complexity to manage the merging phase
    // hides the benefit of it
    disable_lazy_write();

    TupleRowFactory * v_single_factory = new TupleRowFactory(table_metadata->get_single_value(value_name));
    const TupleRow *k = k_factory->make_tuple(keys);
    const TupleRow *v = v_single_factory->make_tuple(values);
    this->write_to_cassandra(k, v);
    delete (v_single_factory);
    delete (k);
    delete (v);
}

/* Returns True if there is still work to do */
bool Writer::call_async() {

    //current write data
    std::pair<const TupleRow *, const TupleRow *> item;
    ncallbacks++; // Increase BEFORE try_pop to avoid race at 'wait_writes_completion'
    if (!data.try_pop(item)) {
        ncallbacks--;
        return false;
    }

    async_query_execute(item.first, item.second);

    return true;
}

