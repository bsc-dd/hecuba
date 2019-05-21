#ifndef HFETCH_TIMEGEN_H
#define HFETCH_TIMEGEN_H

#include <iostream>
#include <chrono>

#include "ModuleException.h"

/***
 * Generates a monotonic strictly increasing timestamp.
 */

class TimestampGenerator {
public:

    TimestampGenerator() = default;

    int64_t next();

private:

    int64_t last;
};


#endif //HFETCH_TIMEGEN_H