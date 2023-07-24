#include "WriterThread.h"
#include "HecubaExtrae.h"
#include <sys/wait.h>

#ifndef CLONE
#include <pthread.h>
#endif


#define DEFAULT_WRITER_CALLBACKS 16
#define DEFAULT_WRITER_BUFF 1000

WriterThread& WriterThread::get(std::map<std::string, std::string>&config) {
    static WriterThread currentWriterThread = { config };
    return currentWriterThread;
}

WriterThread::WriterThread(std::map<std::string, std::string>& config):
    sempending_data(new Semaphore(0)),
    ncallbacks(0),
    error_count(0),
    msgid(0),
    finish_async_query_thread(false)
{
    HecubaExtrae_event(HECUBADBG, HECUBA_CREATEASYNCTHREAD);
    int32_t buff_size = DEFAULT_WRITER_BUFF;
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
    this->data.set_capacity(buff_size);

    int32_t max_callbacks = DEFAULT_WRITER_CALLBACKS;
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
    this->max_calls = (uint32_t) max_callbacks;

    semmaxcallbacks = new Semaphore(max_callbacks);
    create_working_threads();

    HecubaExtrae_event(HECUBADBG, HECUBA_END);
}

/* create_working_threads: Create the pool of threads to send data to the cassandra driver */
void WriterThread::create_working_threads() {
    async_query_thread = std::thread(&WriterThread::async_query_thread_code, this);
    /** 
     * NOTE: This function uses 'std::thread' to create the working threads,
     * but at least on Linux this uses pthread_create which takes about 20ms to
     * finish, this is at least 3 orders of magnitude greater than doing the
     * same with the Linux clone syscall (commented below)
     *
     * char* pila=(char*) malloc(4096);
     * async_query_threadpid = clone(&WriterThread::async_query_thread_code_for_clone, &pila[4096], CLONE_VM, this);
     *
     */
}

// wait for callbacks execution for all sent write requests
void WriterThread::wait_writes_completion(void) {
    HecubaExtrae_event(HECUBADBG, HECUBA_FLUSHELEMENTS);
    //std::cout<< "Writer::wait_writes_completion * Waiting for "<< data.size() << " Pending "<<ncallbacks<<" callbacks" <<" inflight"<<std::endl;
    while(!data.empty() || ncallbacks>0) {
        std::this_thread::yield();
    }
    HecubaExtrae_event(HECUBADBG, HECUBA_END);
}

WriterThread::~WriterThread() {
    // wait for remaining callbacks
    wait_writes_completion();
    // Finish thread
    this->finish_async_query_thread = true; // Mark the async thread to finish BEFORE unblocking it.
    sempending_data->release();// Unblock the async_query_thread (which does not have any work)
    this->async_query_thread.join();
    delete(sempending_data);
    delete(semmaxcallbacks);
}


/* Queue a new pair {keys, values} into the 'data' queue to be executed later.
 * Args are copied, therefore they may be deleted after calling this method. */
void WriterThread::queue_async_query( const Writer* w, const TupleRow *keys, const TupleRow *values) {
    std::tuple<const Writer*, const TupleRow *, const TupleRow *> item = std::make_tuple(w, keys, new TupleRow(values));

    //std::cout<< "  Writer::flushing item created pair"<<std::endl;
    data.push(item);
    sempending_data->release(); //One more pending msg
}

void WriterThread::callback(CassFuture *future, void *ptr) {
    void **data = reinterpret_cast<void **>(ptr);
    assert(data != NULL && data[0] != NULL);
    WriterThread *WThread = (WriterThread *) data[0];
    WThread->semmaxcallbacks->release(); // Limit number of callbacks

    //std::cout<< "Writer::callback"<< std::endl;
    CassError rc = cass_future_error_code(future);
    if (rc != CASS_OK) {
        std::string message(cass_error_desc(rc));
        const char *dmsg;
        size_t l;
        cass_future_error_message(future, &dmsg, &l);
        std::string msg2(dmsg, l);
        WThread->set_error_occurred("Writer callback: " + message + "  " + msg2, data[1], data[2], data[3]);
    } else {
        delete ((TupleRow *) data[2]);
        delete ((TupleRow *) data[3]);
        WThread->ncallbacks--;
        ((Writer*) data[1])->finish_async_call(); //Notify Writer of another finished request.
    }
    HecubaExtrae_comm(EXTRAE_USER_RECV, (long long int)data[4]);
    free(data);
}

void WriterThread::async_query_execute(const Writer* w, const TupleRow *keys, const TupleRow *values) {

    CassStatement *statement = w->bind_cassstatement(keys, values);

    semmaxcallbacks->acquire(); // Limit number of callbacks

    HecubaExtrae_event(HECUBACASS, HBCASS_SENDDRIVER);
#ifdef EXTRAE
    const void **data = (const void **) malloc(sizeof(void *) * 5);
#else
    const void **data = (const void **) malloc(sizeof(void *) * 4);
#endif
    data[0] = this;
    data[1] = w;
    data[2] = keys;
    data[3] = values;
#ifdef EXTRAE
    msgid++;
    data[4] = (void*)((((long long int)getpid())<<32) | msgid);

    HecubaExtrae_comm(EXTRAE_USER_SEND, (long long int)data[4]); // parameter is used to  identify the callback (lower 12 bits from data will be zeroed and then the 12 lower bits from PID added)
#endif /* EXTRAE */
    CassFuture *query_future = cass_session_execute(w->get_session(), statement);
    HecubaExtrae_event(HECUBACASS, HBCASS_END);


    cass_statement_free(statement);


    cass_future_set_callback(query_future, callback, data);
    cass_future_free(query_future);
}

void WriterThread::set_error_occurred(std::string error, const void* writer_p, const void *keys_p, const void *values_p) {
    ++error_count;

    if (error_count > MAX_ERRORS) {
        --ncallbacks;
        throw ModuleException("Try # " + std::to_string(MAX_ERRORS) + " :" + error);
    } else {
        std::cerr << "Connectivity problems: " << error_count << " (" << error << std::endl;
        std::cerr << "  WARNING: We can NOT ensure write requests (table: " << ((Writer *)writer_p)->get_metadata()->get_table_name() << ") order->POTENTIAL INCONSISTENCY"<<std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    const Writer *w = (Writer*) writer_p;
    const TupleRow *keys = (TupleRow *) keys_p;
    const TupleRow *values = (TupleRow *) values_p;

    /** write the data which hasn't been written successfully **/
    async_query_execute(w, keys, values);
}

/* Returns True if there is still work to do */
bool WriterThread::call_async() {

    //current write data
    std::tuple<const Writer*, const TupleRow *, const TupleRow *> item;
    ncallbacks++; // Increase BEFORE try_pop to avoid race at 'wait_writes_completion'
    if (!data.try_pop(item)) {
        ncallbacks--;
        return false;
    }

    async_query_execute(std::get<0>(item), std::get<1>(item), std::get<2>(item));

    return true;
}

void WriterThread::async_query_thread_code() 
{
    while(!finish_async_query_thread) {
        //std::cout<< "Writer::async_query_thread_code "<< std::this_thread::get_id() << " waits..." << std::endl;
        sempending_data->acquire(); // Wait for pending data
        //std::cout<< "Writer::async_query_thread_code "<< std::this_thread::get_id() << " awakes..." << std::endl;
        HecubaExtrae_event(HECUBATHREADASYNC, 1);
        call_async();
        HecubaExtrae_event(HECUBATHREADASYNC, 0);
    }
}
//int WriterThread::async_query_thread_code_for_clone(void* p) {
//    WriterThread * esto = (WriterThread*)p;
//    esto->async_query_thread_code();
//}
