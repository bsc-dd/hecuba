#ifndef HFETCH_METADATA_H
#define HFETCH_METADATA_H

#include "ModuleException.h"
#include <cassandra.h>
#include <cstdint>
#include <algorithm>

struct ColumnMeta {
    uint16_t position;
    CassValueType type;
    std::string name;
};


#endif //HFETCH_METADATA_H
