#ifndef HFETCH_TIMEGEN_H
#define HFETCH_TIMEGEN_H

#include <iostream>
#include <chrono>

#include "ModuleException.h"

#include <mutex>

/***
 * Generates a monotonic strictly increasing timestamp.
 */

class TimestampGenerator {
public:

    TimestampGenerator() = default;

    int64_t next();

private:

    int64_t last;
    std::mutex m;
};


#endif //HFETCH_TIMEGEN_H
