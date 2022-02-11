#ifndef PREFETCHER_MY_TUPLE_FACTORY_H
#define PREFETCHER_MY_TUPLE_FACTORY_H


#define CHECK_CASS(msg) if(rc != CASS_OK && rc != CASS_ERROR_LIB_NULL_VALUE){ \
std::string error(cass_error_desc(rc));\
throw ModuleException(error + ". " + msg);};\


#include <cassert>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <cassandra.h>
#include <stdexcept>
#include <memory>
#include <stdlib.h>

#include "TableMetadata.h"
#include "ModuleException.h"
#include "TupleRow.h"


class TupleRowFactory {

public:
    TupleRowFactory(std::shared_ptr<const std::vector<ColumnMeta> > row_info);

    //Used to pass TupleRowFactory by reference
    TupleRowFactory() {};

    ~TupleRowFactory() {}

    TupleRow *make_tuple(const CassRow *row);

    TupleRow *make_tuple(void *data);


    void bind(CassStatement *statement, const TupleRow *row, u_int16_t offset) const;

    void bind(CassTuple *tuple, const TupleRow *row) const;

    inline std::shared_ptr<const std::vector<ColumnMeta>> get_metadata() const {
        return metadata;
    }

    inline const uint16_t n_elements() const {
        return (uint16_t)
                this->metadata->size();
    }

    inline const uint16_t get_nbytes() const {
        return total_bytes;
    }

private:
    std::shared_ptr<const std::vector<ColumnMeta> > metadata;

    uint16_t total_bytes;

    int cass_to_c(const CassValue *lhs, void *data, int16_t col) const;
    void uuid2cassuuid(const uint64_t** uuid, CassUuid& cass_uuid) const;
    void cassuuid2uuid(const CassUuid& cass_uuid, char** uuid) const ;

};


#endif //PREFETCHER_MY_TUPLE_FACTORY_H
