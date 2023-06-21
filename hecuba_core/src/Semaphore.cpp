#include "Semaphore.h"

Semaphore::Semaphore(int value){
    counter = value;
}

void
Semaphore::release() {
    {
        std::lock_guard<decltype(mx)> lock{mx};
        counter++;
        if (counter <= 0) {
            return;
        }
    }
    cv.notify_one();
}

void
Semaphore::acquire () {
    std::unique_lock<decltype(mx)> lock{mx};
    cv.wait(lock , [&](){return counter > 0; });
    counter--;
}

