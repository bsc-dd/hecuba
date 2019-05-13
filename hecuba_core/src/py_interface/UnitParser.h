#ifndef HFETCH_UNITPARSER_H
#define HFETCH_UNITPARSER_H

#include <Python.h>

#include <cassert>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <cassandra.h>
#include <stdexcept>
#include <memory>
#include <stdlib.h>


#include "../TupleRow.h"
#include "../ModuleException.h"
#include "../TableMetadata.h"
#include "../StorageInterface.h"
#include "NumpyStorage.h"

// Python flags describing the data types
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

class UnitParser {
public:
    UnitParser(const ColumnMeta &CM) {}

    virtual ~UnitParser() {};

    /***
     *
     * @param element Python Object to be transformed to C++
     * @param payload Pointer where the element should be placed once translated
     * @return 0 if successful, -1 if the element is a Py_None, -2 on error
     */
    virtual int16_t py_to_c(PyObject *element, void *payload) const;

    /***
     * @param payload Element to be transformed to a Python Object
     * @return Python Object representing the element pointed by payload represented as the type in Cassandra
     * @throw ModuleException if Python can't parse the object or does not match the Cassandra type
     */
    virtual PyObject *c_to_py(const void *payload) const;

    void error_parsing(std::string type, PyObject *obj) const {
        std::string error_message;
        PyObject *repr = PyObject_Str(obj);
        Py_ssize_t l_size;
        char *l_temp = PyUnicode_AsUTF8AndSize(repr, &l_size);

        if (l_temp != nullptr) {
            error_message = "Parse from python to c, found sth that can't be represented nor parsed";
        } else
            error_message = "Parse from python to c, expected data type " + type +
                            " but the value found is " + std::string(l_temp, (size_t) l_size);
        if (repr && repr->ob_type) error_message += " with type " + std::string(repr->ob_type->tp_name);

        throw TypeErrorException(error_message);
    }
};


class BoolParser : public UnitParser {
public:
    BoolParser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *myint, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;
};


class Int8Parser : public UnitParser {
public:
    Int8Parser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *myint, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;
};


class Int16Parser : public UnitParser {
public:
    Int16Parser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *myint, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;
};


class Int32Parser : public UnitParser {
public:
    Int32Parser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *myint, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;
};


class Int64Parser : public UnitParser {
public:
    Int64Parser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *myint, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;
};


class DoubleParser : public UnitParser {
public:
    DoubleParser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *myint, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;

private:
    bool isFloat;
};


class TextParser : public UnitParser {
public:
    TextParser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *text, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;
};


class BytesParser : public UnitParser {
public:
    BytesParser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *text, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;
};


class UuidParser : public UnitParser {
public:
    UuidParser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *text, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;
};

class TupleParser : public UnitParser {

public:

    TupleParser(const ColumnMeta &CM);

    virtual int16_t py_to_c(PyObject *text, void *payload) const;

    virtual PyObject *c_to_py(const void *payload) const;

private:
    ColumnMeta col_meta;

};


#endif //HFETCH_UNITPARSER_H
