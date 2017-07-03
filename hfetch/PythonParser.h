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

class PythonParser{

public:
    PythonParser(std::shared_ptr<const std::vector<ColumnMeta> > metadatas);

    ~PythonParser();

    TupleRow* make_tuple(PyObject* obj) const;

    PyObject* make_pylist(std::vector<const TupleRow *> &values) const;

private:
    class InnerParser{
    public:
        InnerParser(){};//to be removed
        InnerParser(const ColumnMeta& CM) {}
        virtual ~InnerParser() {};
        virtual int16_t py_to_c(PyObject* element,void* payload) const;
        virtual PyObject* c_to_py(const void* payload) const;

        void error_parsing(std::string type, PyObject* obj) const{
            char *l_temp;
            Py_ssize_t l_size;
            PyObject* repr = PyObject_Str(obj);
            int ok = PyString_AsStringAndSize(repr, &l_temp, &l_size);
            if (ok<0)
                throw TypeErrorException("Parse from python to c, found sth that can't be represented nor parsed");
            throw TypeErrorException("Parse from python to c, expected "+type+", found: "+std::string(l_temp,(size_t)l_size));
        }

    private:
    };


    class TextParser: public InnerParser {

    public:
        TextParser(const ColumnMeta& CM):InnerParser(CM){
            if (CM.size!=sizeof(char*))
                throw ModuleException("Bad size allocated for a text");
        }

        virtual int16_t py_to_c(PyObject* text, void* payload) const {
            if (text==Py_None) return -1;
            if (PyString_Check(text)) {
                char *l_temp;
                Py_ssize_t l_size;
                // PyString_AsStringAndSize returns internal string buffer of obj, not a copy.
                if (PyString_AsStringAndSize(text, &l_temp, &l_size) < 0) error_parsing("PyString",text);
                char *permanent = (char *) malloc(l_size + 1);
                memcpy(permanent, l_temp, l_size);
                permanent[l_size] = '\0';
                memcpy(payload, &permanent, sizeof(char *));
                return 0;
            }
            error_parsing("PyString",text);
            return -2;
        }

        PyObject* c_to_py(const void* payload) const {
            if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to txtptr, found NULL");
            int64_t  *addr = (int64_t*) ((char*)payload);
            char *d = reinterpret_cast<char *>(*addr);
            if (d == nullptr) throw ModuleException("Error parsing from C to Py, expected ptr to text, found NULL");
            return PyUnicode_FromString(d);
        }
    private:
    };


    class Int32Parser: public InnerParser {

    public:
        Int32Parser(const ColumnMeta& CM):InnerParser(CM){
            if (CM.size!=sizeof(int32_t))
                throw ModuleException("Bad size allocated for a Int32");
        }

        virtual int16_t py_to_c(PyObject* myint, void* payload) const {
            if (myint==Py_None)
                return -1;

            if (PyInt_Check(myint)) {
                int32_t t; //TODO it might be safe to pass the payload instead of the var t
                if (PyArg_Parse(myint, Py_INT, &t)<0) error_parsing("PyInt32",myint);
                memcpy(payload, &t, sizeof(t));
                return 0;
            }
            error_parsing("PyInt32",myint);
            return -2;
        }

        PyObject* c_to_py(const void* payload) const {
            if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int, found NULL");
            const int32_t *temp = reinterpret_cast<const int32_t *>(payload);
            try {
                return Py_BuildValue(Py_INT, *temp);
            }
            catch(std::exception &e) {
                throw ModuleException("Error parsing from C to Py, expected Int "+std::string(e.what()));
            }
        }

    private:
    };
    std::vector<InnerParser*> parsers;
    std::shared_ptr<const std::vector<ColumnMeta> > metas; //TODO To be removed
};


#endif //PYTHON_PARSER_H
