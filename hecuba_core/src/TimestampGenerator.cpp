#include "TimestampGenerator.h"


using clock_type = std::chrono::steady_clock;

int64_t TimestampGenerator::next() {


    auto tse = clock_type::now().time_since_epoch();

    int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(tse).count();

    m.lock();
    if (now <= last) now = last + 1;

    last = now;
    m.unlock();

    return last;
}
