#ifndef PREFETCHER_MY_TUPLE_FACTORY_H
#define PREFETCHER_MY_TUPLE_FACTORY_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


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
#include <numpy/arrayobject.h>

//#include <numpy/ndarraytypes.h>
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
    TupleRowFactory(const CassTableMeta *table_meta, const std::vector< std::vector<std::string> > &col_names);

    //Used to pass TupleRowFactory by reference
    TupleRowFactory(){};

    ~TupleRowFactory() {}

    TupleRow* make_tuple(PyObject* obj);

    TupleRow* make_tuple(const CassRow* row);

    PyObject* tuples_as_py(std::vector<const TupleRow *>& values) const;

    std::vector<void*> split_array(PyObject *value);

    PyObject* merge_blocks_as_nparray(std::vector<const TupleRow*>& blocks) const;


    std::vector<const TupleRow*> blocks_to_tuple(std::vector <void*>& blocks, PyObject *obj) const ;
    void* extract_array(PyObject *obj) const;
    std::vector<const TupleRow*> make_tuples_with_npy(PyObject *obj);

    void bind(CassStatement *statement,const  TupleRow *row,  u_int16_t offset) const;

    inline uint16_t n_elements(){
        return this->metadata.size();
    }

    inline RowMetadata get_metadata() const{
        return metadata;

    }

private:
    RowMetadata metadata;

    uint16_t total_bytes;

    uint16_t compute_size_of(const CassValueType VT) const;

    PyObject* c_to_py(const void *V,const ColumnMeta &meta) const;

    int py_to_c(PyObject *obj, void* data, int32_t col) const;

    int cass_to_c(const CassValue *lhs,void * data, int16_t col) const;

};


#endif //PREFETCHER_MY_TUPLE_FACTORY_H
