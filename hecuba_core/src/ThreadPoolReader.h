#ifndef THREADPOOLREADER_THREADPOOLREADER_H
#define THREADPOOLREADER_THREADPOOLREADER_H

#include <thread>
#include <queue>

#include "Semaphore.h"

#include "tbb/concurrent_queue.h"
//#include "tbb/concurrent_hash_map.h"

class ThreadPoolReader {
public:
    //ThreadPoolReader(int nThreads);
    ThreadPoolReader(ThreadPoolReader const&)   = delete;
    void operator=(ThreadPoolReader const&) = delete;
    static ThreadPoolReader* getInstance(int n);

    int getNumThreads();

    void queueJob(int x);
    void queueJobs(std::vector<int> v);
    void queueJobAndWait(int x);
    void queueJobsAndWait(const uint8_t* src, char* dst, int n_threads, int file_size);

private:

    struct jobInfo {
        int id;
        int n_threads;
        int file_size;
        const uint8_t* src;
        char* dst;
    };

    ThreadPoolReader(int nThreads);
    ~ThreadPoolReader();

    void thread_code();
    bool call_async();
    void executeJob(jobInfo info);
    void create_working_threads(int n);

    static ThreadPoolReader* threadPoolReader;
    bool resume_queue = true;
    std::vector<std::thread> threads;
    Semaphore* semQueueJob;
    Semaphore* semResultJob;
    tbb::concurrent_bounded_queue<jobInfo> qJobs;
};

#endif //THREADPOOLREADER_THREADPOOLREADER_H
