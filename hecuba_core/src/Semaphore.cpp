#include "Semaphore.h"

Semaphore::Semaphore(int value){
    counter = value;
}

Semaphore::Semaphore(int value, int limit){
    counter = value;
    threshold = limit;
}

void
Semaphore::release(int update) {
    {
        std::lock_guard<decltype(mx)> lock{mx};
        counter += update;
        if (counter <= 0) {
            return;
        }
    }
    //cv.notify_one(); //TODO check difference between notify_one and notify_all
    cv.notify_all();
}

void
Semaphore::acquire () {
    std::unique_lock<decltype(mx)> lock{mx};
    cv.wait(lock , [&](){return counter > 0; });
    counter--;
}

void
Semaphore::acquire_many () {
    std::unique_lock<decltype(mx)> lock{mx};
    cv.wait(lock , [&](){return counter >= threshold; });
    counter -= threshold;
}

void
Semaphore::set_threshold(int limit) {
    threshold = limit;
}