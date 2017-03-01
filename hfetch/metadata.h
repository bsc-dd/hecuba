//
// Created by ccugnasc on 2/28/17.
//

#ifndef HFETCH_METADATA_H
#define HFETCH_METADATA_H


#include <cassandra.h>
#include <cstdint>
#include <map>
#include <numpy/arrayobject.h>

struct ColumnMeta {
    uint16_t position;
    CassValueType type;
    std::vector<std::string> info;

    NPY_TYPES get_arr_type() {
        if (info.size() != 3) return NPY_NOTYPE;
        if (info[1] == "bool")
            return NPY_BOOL;
        if (info[1] == "byte")
            return NPY_BYTE;
        if (info[1] == "ubyte")
            return NPY_UBYTE;
        if (info[1] == "short")
            return NPY_SHORT;
        if (info[1] == "ushort")
            return NPY_USHORT;
        if (info[1] == "int")
            return NPY_INT;
        if (info[1] == "uint")
            return NPY_UINT;
        if (info[1] == "long")
            return NPY_LONG;
        if (info[1] == "ulong")
            return NPY_ULONG;
        if (info[1] == "longlong")
            return NPY_LONGLONG;
        if (info[1] == "float")
            return NPY_FLOAT;
        if (info[1] == "double")
            return NPY_DOUBLE;
        if (info[1] == "clongdouble")
            return NPY_LONGDOUBLE;
        if (info[1] == "cfloat")
            return NPY_CFLOAT;
        if (info[1] == "cdouble")
            return NPY_CDOUBLE;
        if (info[1] == "clongdouble")
            return NPY_CLONGDOUBLE;
        if (info[1] == "obj")
            return NPY_OBJECT;
        if (info[1] == "str")
            return NPY_STRING;
        if (info[1] == "unicode")
            return NPY_UNICODE;
        if (info[1] == "void")
            return NPY_VOID;
        return NPY_NOTYPE;
    }

    PyObject *get_arr_dims() {
        if (info.size() != 3)
            return Py_None;
        PyObject *dims = PyList_New(0);
        std::string temp = info[2];
        int pos = 0;
        while ((pos = temp.find('x')) != temp.npos) {
            PyList_Append(dims, PyInt_FromLong(std::atoi(temp.substr(0, pos).c_str())));
            temp = temp.substr(pos + 1, temp.size());
            //we assume pos(x)+1 <= dimensions length
        }
        PyList_Append(dims, PyInt_FromLong(std::atoi(temp.substr(0, 1).c_str())));
        return dims;
    }

};


#endif //HFETCH_METADATA_H
