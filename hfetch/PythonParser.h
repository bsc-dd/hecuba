#ifndef PYTHON_PARSER_H
#define PYTHON_PARSER_H

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
#include "TableMetadata.h"
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


class PythonParser{

public:
    PythonParser();

    ~PythonParser();

    TupleRow* make_tuple(PyObject* obj,std::shared_ptr<const std::vector<ColumnMeta> > metadata) const;

    PyObject* tuple_as_py(const TupleRow* tuple, std::shared_ptr<const std::vector<ColumnMeta> > metadata) const;


private:

    PyObject* c_to_py(const void *V, ColumnMeta &meta) const;

    int py_to_c(PyObject *obj, void* data, int32_t col) const;


};


#endif //PYTHON_PARSER_H
