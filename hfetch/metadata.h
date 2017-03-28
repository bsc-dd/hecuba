#ifndef HFETCH_METADATA_H
#define HFETCH_METADATA_H

#include "ModuleException.h"
#include <cassandra.h>
#include <cstdint>
#include <algorithm>
#include <numpy/arrayobject.h>


struct ColumnMeta {
    uint16_t position;
    CassValueType type;
    std::vector<std::string> info;

    const NPY_TYPES get_arr_type() const {
        if (info.size() != 4) {
            throw ModuleException("Numpy array metadata must consist of [name,type,dimensions,partition_option]");//return NPY_NOTYPE;
        }
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

    PyArray_Dims *get_arr_dims() const {
        if (info.size() != 4)
            throw ModuleException("Numpy array metadata must consist of [name,type,dimensions,partition]");

        std::string temp = info[2];

        ssize_t n = std::count(temp.begin(), temp.end(), 'x') + 1;

        npy_intp* ptr = new npy_intp[n];//[n];

        size_t pos = 0;
        uint16_t i = 0;
        while ((pos = temp.find('x')) != temp.npos) {
            if (i>n) throw ModuleException("Bad formed dimensions array");
            ptr[i]=std::atoi(temp.substr(0, pos).c_str());
            temp = temp.substr(pos + 1, temp.size());
            //we assume pos(x)+1 <= dimensions length
            ++i;
        }
        ptr[i]=std::atoi(temp.c_str());
        PyArray_Dims *dims = new PyArray_Dims{ptr,(int)n};
        return dims;
    }
};

struct RowMetadata {
    bool has_numpy = false;
    std::shared_ptr<std::vector<ColumnMeta> > cols_meta;

    RowMetadata(){}
    RowMetadata( std::shared_ptr<std::vector<ColumnMeta> > metas) {
        this->cols_meta=metas;
    }

    uint16_t size() const {
        return (uint16_t) cols_meta->size();
    }

    const ColumnMeta at(uint16_t pos) const{
        return cols_meta.get()->at(pos);
    }
};


#endif //HFETCH_METADATA_H
