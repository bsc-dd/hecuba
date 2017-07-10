#ifndef PYTHON_PARSER_H
#define PYTHON_PARSER_H

#include <python2.7/Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include "numpy/arrayobject.h"

#include <cassert>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <cassandra.h>
#include <stdexcept>
#include <memory>
#include <stdlib.h>


#include "TupleRow.h"
#include "ModuleException.h"
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

//bytes
#define maxarray_size 65536

class PythonParser {

public:
    PythonParser();

    ~PythonParser();

    TupleRow *make_tuple(PyObject *obj, std::shared_ptr<const std::vector <ColumnMeta> > metadata) const;

    PyObject *merge_blocks_as_nparray(std::vector<const TupleRow *> &blocks,
                                      std::shared_ptr<const std::vector <ColumnMeta> > metadata) const;

    PyObject *tuples_as_py(std::vector<const TupleRow *> &values,
                           std::shared_ptr<const std::vector <ColumnMeta> > metadata) const;

    std::vector<void *> split_array(PyObject *py_array);

    std::vector<const TupleRow *>
    make_tuples_with_npy(PyObject *obj, std::shared_ptr<const std::vector <ColumnMeta> > metadata);

    void *extract_array(PyObject *py_array) const;

    std::vector<const TupleRow *>
    blocks_to_tuple(std::vector<void *> &blocks, std::shared_ptr<const std::vector <ColumnMeta> > metadata,
                    PyObject *obj) const;

    const NPY_TYPES get_arr_type(const ColumnMeta &column_meta) const;

    PyArray_Dims *get_arr_dims(const ColumnMeta &column_meta) const;

private:

    PyObject *c_to_py(const void *V, const ColumnMeta &meta) const;

    int py_to_c(PyObject *obj, void *data, CassValueType type) const;

};


#endif //PYTHON_PARSER_H
