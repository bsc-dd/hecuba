#ifndef __WRITER_THREAD_H__
#define __WRITER_THREAD_H__

#include <thread>
#include <string>
#include <map>
#include <cstdint>
#include "Writer.h"
#include "TupleRow.h"
#include "Semaphore.h"

namespace Hecuba {
#define MAX_ERRORS 10

#define CLONE

class WriterThread;

struct callback_type {  // Structure used to pass information in cassandra callback
        WriterThread*   writerTH;       //'this'
        Writer*         w;
        const TupleRow* keys;
        const TupleRow* values;
        uint64_t        retries;
#ifdef EXTRAE
        uint32_t        msgid;
        long long int   start_time;
#endif /* EXTRAE */
};

class WriterThread {
    public:
        static WriterThread& get(std::map<std::string, std::string>& config);
        WriterThread(WriterThread const&)   = delete;
        void operator=(WriterThread const&) = delete;
        void queue_async_query( Writer* w, const TupleRow *keys, const TupleRow *values);

        static int async_query_thread_code_for_clone(void*);
        static void* async_query_thread_code_for_pthread_create(void*);
    private:
        WriterThread(std::map<std::string, std::string>& config);
        ~WriterThread();
        bool call_async();
        void async_query_thread_code();
        void set_error_occurred(std::string error, struct callback_type* data);
        static void callback(CassFuture *future, void *ptr);
        void async_query_execute(Writer* w, const TupleRow *keys, const TupleRow *values);
        void async_query_execute(struct callback_type *data);
        void wait_writes_completion(void);
        void create_working_threads(void);

        bool finish_async_query_thread = false;
        std::thread async_query_thread;
        //int async_query_threadpid = -1;

        Semaphore* sempending_data;  // Synchronization semaphore to wait for new elements in 'data'
        Semaphore* semmaxcallbacks; //Resource limiting Semaphore to limit the number of in_flight callbacks.
        uint32_t max_calls;
        std::atomic<uint32_t> ncallbacks;
#ifdef EXTRAE
        std::atomic<uint32_t> msgid;
#endif /*EXTRAE*/

        tbb::concurrent_bounded_queue <std::tuple<Writer*, const TupleRow *, const TupleRow *>> data;
        uint64_t data_close_to_max_times=0;      //Number of times the buffer was almost full (data.size() == (data.capacity()-1))

};
};
#endif /* __WRITER_THREAD_H__ */
