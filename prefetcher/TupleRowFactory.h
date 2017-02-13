//
// Created by bscuser on 1/26/17.
//

#ifndef PREFETCHER_MY_TUPLE_FACTORY_H
#define PREFETCHER_MY_TUPLE_FACTORY_H

#include <cassert>


#include <cstring>
#include <string>
#include <iostream>

#include <vector>
#include <cassandra.h>
#include <python2.7/Python.h>

#include "TupleRow.h"
#include <stdexcept>

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
    }

    TupleRow* make_tuple(PyObject* obj);

    TupleRow* make_tuple(const CassRow* row);

    PyObject* tuple_as_py(TupleRow* tuple) const;

    inline CassValueType get_type(uint16_t pos) {
        return type_array[pos];
    }

    inline uint16_t n_elements(){
        return (uint16_t) this->type_array.size();
    }
    TupleRowFactory(){};

private:
    std::vector<uint16_t> offsets;
    std::vector<CassValueType> type_array;
    std::vector<std::string> name_map;
    uint16_t total_bytes;

    uint16_t compute_size_of(const CassValueType VT) const;

    PyObject* c_to_py(const void *V, CassValueType VT) const;

    int py_to_c(PyObject *key, void* data, int32_t col) const;

    int cass_to_c(const CassValue *lhs,void * data, int16_t col) const;

};


#endif //PREFETCHER_MY_TUPLE_FACTORY_H
