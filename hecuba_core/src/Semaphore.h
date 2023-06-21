#ifndef _SEMAPHORE_H__
#define _SEMAPHORE_H__

#include <mutex>
#include <condition_variable>

class Semaphore {
public:

    Semaphore(int value);
    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;
    void release();
    void acquire();

private:

    int counter{0};
    std::condition_variable cv;
    std::mutex mx;

};
#endif
