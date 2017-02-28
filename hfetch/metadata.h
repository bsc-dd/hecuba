//
// Created by ccugnasc on 2/28/17.
//

#ifndef HFETCH_METADATA_H
#define HFETCH_METADATA_H


#include <cassandra.h>
#include <cstdint>

struct ColumnMeta{
    uint16_t position;
    CassValueType type;
    std::string name;
};


#endif //HFETCH_METADATA_H
