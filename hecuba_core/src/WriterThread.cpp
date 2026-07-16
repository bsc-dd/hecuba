#include "WriterThread.h"
#include "HecubaExtrae.h"
#include "debug.h"
#include <sys/wait.h>

#ifndef CLONE
#include <pthread.h>
#endif
namespace Hecuba {


#define DEFAULT_WRITER_CALLBACKS 16
#define DEFAULT_WRITER_BUFF 1000

WriterThread& WriterThread::get(std::map<std::string, std::string>&config) {
    static WriterThread currentWriterThread = { config };
    return currentWriterThread;
}

WriterThread::WriterThread(std::map<std::string, std::string>& config):
    sempending_data(new Semaphore(0)),
    ncallbacks(0),
#ifdef EXTRAE
    msgid(0),
#endif /* EXTRAE */
    finish_async_query_thread(false)
{
    HecubaExtrae_event(HECUBADBG, HECUBA_CREATEASYNCTHREAD);
    int32_t buff_size = DEFAULT_WRITER_BUFF;
    if (config.find("write_buffer_size") != config.end()) {
        std::string buff_size_str = config["write_buffer_size"];
        try {
            buff_size = std::stoi(buff_size_str);
            if (buff_size < 0) throw ModuleException("Writer buffer value must be >= 0");
        }
        catch (std::exception &e) {
            std::string msg(e.what());
            msg += " Malformed value in config for write_buffer_size";
            throw ModuleException(msg);
        }
    }
    this->data.set_capacity(buff_size);

    int32_t max_callbacks = DEFAULT_WRITER_CALLBACKS;
    if (config.find("write_callbacks_number") != config.end()) {
        std::string max_callbacks_str = config["write_callbacks_number"];
        try {
            max_callbacks = std::stoi(max_callbacks_str);
            if (max_callbacks <= 0) throw ModuleException("Writer parallelism value must be > 0");
        }
        catch (std::exception &e) {
            std::string msg(e.what());
            msg += " Malformed value in config for write_callbacks_number";
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
    HecubaExtrae_event(HECUBACASS_NCALLBACKS, ncallbacks);
    HecubaExtrae_event(HECUBADBG, HECUBA_FLUSHELEMENTS);
    //std::cout<< "Writer::wait_writes_completion * Waiting for "<< data.size() << " Pending "<<ncallbacks<<" callbacks" <<" inflight"<<std::endl;
    while(!data.empty() || ncallbacks>0) {
        std::this_thread::yield();
    }
    HecubaExtrae_event(HECUBADBG, HECUBA_END);
    HecubaExtrae_event(HECUBACASS_NCALLBACKS, 0);
}

WriterThread::~WriterThread() {
    // wait for remaining callbacks
    wait_writes_completion();
    // Finish thread
    this->finish_async_query_thread = true; // Mark the async thread to finish BEFORE unblocking it.
    sempending_data->release();// Unblock the async_query_thread (which does not have any work)
    this->async_query_thread.join();
    //waitpid(async_query_threadpid, NULL, 0); // TODO: CHECK ERRORS!
    if (data_close_to_max_times>0) {
	    std::cerr<<"WARN: WriterThread::queue_async_query: data capacity was close to "<<data.capacity()<<" "<< data_close_to_max_times<<" times. Maybe increasing WRITE_BUFFER_SIZE is required."<<std::endl;
    }
    delete(sempending_data);
    delete(semmaxcallbacks);
}


/* Queue a new pair {keys, values} into the 'data' queue to be executed later.
 * Args are copied, therefore they may be deleted after calling this method. */
void WriterThread::queue_async_query( Writer* w, const TupleRow *keys, const TupleRow *values) {
    try {
    std::tuple<Writer*, const TupleRow *, const TupleRow *> item = std::make_tuple(w, keys, new TupleRow(values));

    //std::cout<< "  Writer::flushing item created pair"<<std::endl;
#if 1
    if (!data.try_push(item)) { // 'data' BLOCKS thread if full capacity is achieved, therefore yield the CPU to Cassandra (this is useful in the shared scenario when appl and cassandra are sharing nodes
        HecubaExtrae_event(HECUBAFULLBUFFER, 1);
        cpu_set_t app_mask; // Original APPLICATION mask
        int is_remove_needed=0;
        uint32_t userID;
        if (w->getConfigValue(std::string("dynamic_affinity")) == std::string("true")) {
                sched_getaffinity(0, sizeof(app_mask), &app_mask);
                try{
                    userID = HecubaSession::get().getUserID();
                    if (HecubaSession::get().addCassandraAffinity(userID, &app_mask) >=0) is_remove_needed = 1;
                } catch(std::out_of_range e) {
                    std::cerr<<"HecubaSession::getUserID: thread id " << std::this_thread::get_id()<< "is not registered at translation map to user ids" << std::endl;
                }
        }
        data.push(item);
        if (is_remove_needed) HecubaSession::get().removeCassandraAffinity(userID, &app_mask);
        HecubaExtrae_event(HECUBAFULLBUFFER, 0);
    }
#else
    data.push(item);
#endif
    if (data.size() ==  (data.capacity()-1)) {
        data_close_to_max_times ++;
	    //std::cerr<<"WARN: WriterThread::queue_async_query: data capacity is "<<data.size()<<" close to full. Maybe increasing WRITE_BUFFER_SIZE is required."<<std::endl;
    }
    sempending_data->release(); //One more pending msg
    }catch (std::exception &e) {
            std::cerr << "WriterThread.cpp:: queue_async_query " <<std::endl;
            std::cerr << e.what();
            std::cerr << "I am process ID "<<  getpid() << std::endl;
            throw e;
    };

}

void WriterThread::callback(CassFuture *future, void *ptr) {
    struct callback_type *data = reinterpret_cast<struct callback_type*>(ptr);
    DBG("WriterThread::callback");
    assert(data != NULL && data->writerTH != NULL);
    WriterThread *WThread = data->writerTH;
    //WThread->semmaxcallbacks->release(); // Limit number of callbacks

    //std::cout<< "Writer::callback"<< std::endl;
    CassError rc = cass_future_error_code(future);
    if (rc != CASS_OK) {
        DBG("WriterThread::callback. Cassandra returns KO");
        std::string message(cass_error_desc(rc));
        const char *dmsg;
        size_t l;
        cass_future_error_message(future, &dmsg, &l);
        std::string msg2(dmsg, l);
        WThread->set_error_occurred("Writer callback: " + message + "  " + msg2, data);
    } else {
        DBG("WriterThread::callback. Cassandra returns OK");
        delete (data->keys);
        delete (data->values);
        WThread->semmaxcallbacks->release(); // Limit number of callbacks. Release the semaphore only when the query ends ok
        WThread->ncallbacks--;
        data->w->finish_async_call(); //Notify Writer of another finished request.
#ifdef EXTRAE
    struct timespec t2;
    clock_gettime(CLOCK_REALTIME, &t2);
    long long int accum = data->start_time;
    accum = (((long long int)t2.tv_sec)*1000000000L+t2.tv_nsec) - accum;
    HecubaExtrae_event(HECUBACASS_RESPONSETIME, accum);
    HecubaExtrae_event(HECUBACASS_RESPONSETIME, 0);
    if (data->retries > 0)
        HecubaExtrae_event(HECUBACASS_RETRY_NUMBER, data->retries);
#endif /* EXTRAE */
        free(data); // 'data' is only released if successful (otherwise is reused)
    }
    HecubaExtrae_comm(EXTRAE_USER_RECV, (long long int)data->msgid);
}

void WriterThread::async_query_execute(struct callback_type *data) {
    CassStatement *statement = data->w->bind_cassstatement(data->keys, data->values);

    //semmaxcallbacks->acquire(); // Limit number of callbacks. This function is called for the retries by the driver threads. Move the wait of the semaphore to the function that is called for the initial try and do the signal only when the query success

    HecubaExtrae_event(HECUBACASS, HBCASS_SENDDRIVER);
    CassFuture *query_future = cass_session_execute(data->w->get_session(), statement);
    HecubaExtrae_event(HECUBACASS, HBCASS_END);


    cass_statement_free(statement);


    cass_future_set_callback(query_future, callback, (void*)data);
    cass_future_free(query_future);
}

void WriterThread::async_query_execute(Writer* w, const TupleRow *keys, const TupleRow *values) {

    struct callback_type *data = (struct callback_type*) malloc(sizeof(struct callback_type));
    data->writerTH = this;
    data->w = w;
    data->keys = keys;
    data->values = values;
    data->retries = 0; //number of retries
#ifdef EXTRAE
    msgid++;
    data->msgid = ((((unsigned long long int)getpid())<<32) | msgid);
    HecubaExtrae_comm(EXTRAE_USER_SEND, data->msgid); // parameter is used to  identify the callback (lower 12 bits from data will be zeroed and then the 12 lower bits from PID added)
    struct timespec t2;
    clock_gettime(CLOCK_REALTIME, &t2);
    data->start_time = ((unsigned long long int)t2.tv_sec)*1000000000L+t2.tv_nsec;
#endif /* EXTRAE */
    semmaxcallbacks->acquire(); // Limit number of callbacks. Wait here on the semaphore to do it just once per query (not per retry)
    async_query_execute(data);
}

void WriterThread::set_error_occurred(std::string error, struct callback_type* data) {
    if (data->retries > MAX_ERRORS) {
        --ncallbacks;
        throw ModuleException("Try # " + std::to_string(MAX_ERRORS) + " :" + error);
    } else {
        //std::cerr << "Connectivity problems: " << data->retries << " (" << error << std::endl;
        //std::cerr << "  WARNING: We can NOT ensure write requests (table: " << data->w->get_metadata()->get_table_name() << ") order->POTENTIAL INCONSISTENCY"<<std::endl;
        data->retries ++;
    }

    async_query_execute(data);
}

/* Returns True if there is still work to do */
bool WriterThread::call_async() {

    //current write data
    std::tuple<Writer*, const TupleRow *, const TupleRow *> item;
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
}
