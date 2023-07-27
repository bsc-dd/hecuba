#ifndef _SEMAPHORE_H__
#define _SEMAPHORE_H__

#include <mutex>
#include <condition_variable>

class Semaphore {
public:

    Semaphore(int value);
    Semaphore(int value, int limit);
    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;
    void release(int update = 1);
    void acquire();
    void acquire_many();
    void set_threshold(int limit);

private:

    int counter{0};
    std::condition_variable cv;
    std::mutex mx;
    int threshold{1};

};
#endif
