#ifndef __WRITER_THREAD_H__
#define __WRITER_THREAD_H__

#include <thread>
#include <string>
#include <map>
#include "Writer.h"
#include "TupleRow.h"
#include "Semaphore.h"

#define MAX_ERRORS 10

#define CLONE

class WriterThread {
    public:
        static WriterThread& get(std::map<std::string, std::string>& config);
        WriterThread(WriterThread const&)   = delete;
        void operator=(WriterThread const&) = delete;
        void queue_async_query( const Writer* w, const TupleRow *keys, const TupleRow *values);

        static int async_query_thread_code_for_clone(void*);
        static void* async_query_thread_code_for_pthread_create(void*);
    private:
        WriterThread(std::map<std::string, std::string>& config);
        ~WriterThread();
        bool call_async();
        void async_query_thread_code();
        void set_error_occurred(std::string error, const void *writer_p, const void *keys, const void *values);
        static void callback(CassFuture *future, void *ptr);
        void async_query_execute(const Writer* w, const TupleRow *keys, const TupleRow *values);
        void wait_writes_completion(void);
        void create_working_threads(void);

        bool finish_async_query_thread = false;
        std::thread async_query_thread;

        Semaphore* sempending_data;  // Synchronization semaphore to wait for new elements in 'data'
        Semaphore* semmaxcallbacks; //Resource limiting Semaphore to limit the number of in_flight callbacks.
        uint32_t error_count;
        uint32_t max_calls;
        std::atomic<uint32_t> ncallbacks;
#ifdef EXTRAE
        std::atomic<uint32_t> msgid;
#endif /*EXTRAE*/

        tbb::concurrent_bounded_queue <std::tuple<const Writer*, const TupleRow *, const TupleRow *>> data;

};
#endif /* __WRITER_THREAD_H__ */
