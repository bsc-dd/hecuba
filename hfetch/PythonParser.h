#ifndef PYTHON_PARSER_H
#define PYTHON_PARSER_H

#include <python2.7/Python.h>

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
#include "StorageInterface.h"
#include "NumpyStorage.h"


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
    PythonParser(std::shared_ptr<StorageInterface> storage,std::shared_ptr<const std::vector<ColumnMeta> > metadatas);

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

        virtual PyObject* c_to_py(const void* payload) const {
            if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to txtptr, found NULL");
            int64_t  *addr = (int64_t*) ((char*)payload);
            char *d = reinterpret_cast<char *>(*addr);
            if (d == nullptr) throw ModuleException("Error parsing from C to Py, expected ptr to text, found NULL");
            return PyUnicode_FromString(d);
        }
    private:
    };

    class NumpyParser: public InnerParser {
    public:
        NumpyParser(const ColumnMeta& CM):InnerParser(CM){
            if (CM.size!=sizeof(char*))
                throw ModuleException("Bad size allocated for a Int32");
            table=CM.info.at("table");
            attribute_name=CM.info.at("name");
            keyspace=CM.info.at("keyspace");
            //parse storage id
            uint64_t *uuid = (uint64_t*) CM.info.at("storage_id").c_str();
            storage_id = {*uuid,*(uuid+1)};
        }

        virtual int16_t py_to_c(PyObject* numpy, void* payload) const {
            if (numpy==Py_None)
                return -1;
            PyArrayObject *arr;
            int ok = PyArray_OutputConverter(numpy, &arr);
            if (!ok) error_parsing("Numpy",numpy); //failed to convert array from PyObject to PyArray
            np_storage->store(table,keyspace,attribute_name,storage_id,arr);
            return 0;

        }

        virtual PyObject* c_to_py(const void* payload) const {
            if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to bytes, found NULL");
            //np_storage->read()
        }

        void setStorage(std::shared_ptr<StorageInterface> storage) {
            ArrayPartitioner algorithm = ArrayPartitioner();
            np_storage = new NumpyStorage(storage, algorithm);
        }

    private:
        NumpyStorage *np_storage;
        std::string table;
        std::string keyspace;
        CassUuid storage_id;
        std::string attribute_name;

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

        virtual PyObject* c_to_py(const void* payload) const {
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
