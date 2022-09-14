#ifndef UUID_H
#define UUID_H

#include <cstdint>
#include <string>

namespace UUID {
    uint64_t* generateUUID(void) ;
    uint64_t* generateUUID5(const char* name) ;
    std::string UUID2str(const uint64_t* c_uuid);
}

#endif /* UUID_H */
