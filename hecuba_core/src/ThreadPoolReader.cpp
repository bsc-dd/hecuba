//
// Created by enricsosa on 24/07/23.
//
#include "ThreadPoolReader.h"
#include <unistd.h>
#include <cstring>

ThreadPoolReader* ThreadPoolReader::threadPoolReader = nullptr;

ThreadPoolReader* ThreadPoolReader::getInstance(int n) {
    if (threadPoolReader == nullptr) {
        threadPoolReader = new ThreadPoolReader(n);
    }
    return threadPoolReader;
}

ThreadPoolReader::ThreadPoolReader(int nThreads):
resume_queue(true),
semQueueJob(new Semaphore(0)),
semResultJob(new Semaphore(0))
{
    create_working_threads(nThreads);
}

ThreadPoolReader::~ThreadPoolReader() {
    // Finish thread
    this->resume_queue = false; // Mark the async thread to finish BEFORE unblocking it.
    semQueueJob->release();// Unblock the async_query_thread (which does not have any work)
    for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
    delete(semQueueJob);
    delete(semResultJob);
}

void ThreadPoolReader::create_working_threads(int n) {
    //TODO clone instead of std::thread
    threads.resize(n);
    for (int i = 0; i < n; ++i) {
        threads[i] = std::thread(&ThreadPoolReader::thread_code, this);
    }
}

int ThreadPoolReader::getNumThreads() {
    return threads.size();
}

void ThreadPoolReader::queueJob(int x) {
    //qJobs.push(x);
    semQueueJob->release();
}

void ThreadPoolReader::queueJobs(std::vector<int> v) {
    int n = v.size();
    for (int i = 0; i < n; ++i) {
        //qJobs.push(v[i]);
    }
    semQueueJob->release(n);
}

void ThreadPoolReader::queueJobAndWait(int x) {
    //qJobs.push(x);
    semQueueJob->release();
    semResultJob->acquire();
}

void ThreadPoolReader::queueJobsAndWait(const uint8_t* src, char* dst, int n_threads, int file_size) {
    semResultJob->set_threshold(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        jobInfo jobInfo;
        jobInfo.id = i;
        jobInfo.n_threads = n_threads;
        jobInfo.file_size = file_size;
        jobInfo.src = src;
        jobInfo.dst = dst;
        qJobs.push(jobInfo);
    }
    semQueueJob->release(n_threads); //acquire in thread_code()
    semResultJob->acquire_many();
}

void ThreadPoolReader::executeJob(jobInfo info) {
    //TODO last thread may copy a different size of data
    sleep(2);

    int size = info.file_size/info.n_threads;
    int offset = size*info.id / sizeof(uint8_t);

    // memcpy to a mmap region (an optane or ssd file) or malloc region (memory)
    memcpy(&info.dst[offset], &info.src[offset], size);
}

/* Returns True if there is still work to do */
bool ThreadPoolReader::call_async() {
    jobInfo x;
    if (!qJobs.try_pop(x)) {
        return false;
    }
    executeJob(x);
    return true;
}

void ThreadPoolReader::thread_code() {
    while (resume_queue) {
        semQueueJob->acquire();
        call_async();
        semResultJob->release(); //acquire in queueJobsAndWait(...)
    }
}
