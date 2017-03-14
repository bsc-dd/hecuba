#ifndef PREFETCHER_MY_TUPLE_FACTORY_H
#define PREFETCHER_MY_TUPLE_FACTORY_H

#define CHECK_CASS(msg) if(rc != CASS_OK){ \
std::string error(cass_error_desc(rc));\
std::cerr<<msg<<" "<<error<<std::endl; };\

#include <cassert>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <cassandra.h>
#include <python2.7/Python.h>
#include "ModuleException.h"
#include "TupleRow.h"
#include <stdexcept>
#include <memory>
#include <stdlib.h>
#include "metadata.h"

#define Py_STRING "s"
#define Py_U_LONGLONG "K"
#define Py_U_LONG "k"
#define Py_LONGLONG "L"
#define Py_LONG "l"
#define Py_BOOL "b"
#define Py_INT "i"
#define Py_U_INT "I"
#define Py_FLOAT "f"
#define Py_DOUBLE "d"
#define Py_SHORT_INT "h"




class TupleRowFactory{

public:
    TupleRowFactory(const CassTableMeta *table_meta, const std::vector<std::string> &col_names);

    ~TupleRowFactory() {
        metadata = NULL;
    }

    TupleRow* make_tuple(PyObject* obj);

    TupleRow* make_tuple(const CassRow* row);

    PyObject* tuple_as_py(const TupleRow* tuple) const;

    void bind( CassStatement *statement,const  TupleRow *row,  u_int16_t offset) const ;

    inline const CassValueType get_type(uint16_t pos) const {
        return metadata.get()->at(pos).type;
    }

    inline uint16_t n_elements(){
        return (uint16_t) this->metadata.get()->size();
    }
    TupleRowFactory(){};

    std::shared_ptr<std::vector<ColumnMeta>> get_metadata() const{
        return metadata;

    }

private:
    std::shared_ptr<std::vector<ColumnMeta>> metadata;
    uint16_t total_bytes;

    uint16_t compute_size_of(const CassValueType VT) const;

    PyObject* c_to_py(const void *V, CassValueType VT) const;

    int py_to_c(PyObject *key, void* data, int32_t col) const;

    int cass_to_c(const CassValue *lhs,void * data, int16_t col) const;

};


#endif //PREFETCHER_MY_TUPLE_FACTORY_H
