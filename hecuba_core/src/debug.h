#ifndef __DEBUG_H__
#define __DEBUG_H__

#define ENABLE_DEBUG

#ifdef ENABLE_DEBUG
#include <sstream>
#define DBG(x...) \
        do {\
            std::cout<< "DBG " << x << std::endl;\
        } while(0)
#define DBGHEXTOSTRING(_b, _size) \
    do { \
        char *b = (char*)(_b);\
        uint64_t size = (uint64_t) (_size);\
        DBG(" size: "<< size); \
        std::stringstream stream; \
        for (uint64_t i = 0; i<size; i++) { \
            stream << std::hex << int(b[i]); \
        } \
        DBG( stream.str() ); \
    } while(0)
#else
#define DBG(x...)
#define DBGHEXTOSTRING(b,size)
#endif


#endif /* __DEBUG_H__ */
