#include "UnitParser.h"

int16_t UnitParser::py_to_c(PyObject* element,void* payload) const {
    throw ModuleException("Not implemented");
}

PyObject* UnitParser::c_to_py(const void* payload) const {
    throw ModuleException("Not implemented");
}


/*** Int32 parser ***/

Int32Parser::Int32Parser(const ColumnMeta& CM):UnitParser(CM){
    if (CM.size!=sizeof(int32_t))
        throw ModuleException("Bad size allocated for a Int32");
}

int16_t Int32Parser::py_to_c(PyObject* myint, void* payload) const {
    if (myint==Py_None) return -1;

    if (PyInt_Check(myint)) {
        int32_t t; //TODO it might be safe to pass the payload instead of the var t
        if (PyArg_Parse(myint, Py_INT, &t)<0) error_parsing("PyInt32",myint);
        memcpy(payload, &t, sizeof(t));
        return 0;
    }
    error_parsing("PyInt32",myint);
    return -2;
}

PyObject* Int32Parser::c_to_py(const void* payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int, found NULL");
    const int32_t *temp = reinterpret_cast<const int32_t *>(payload);
    try {
        return Py_BuildValue(Py_INT, *temp);
    }
    catch(std::exception &e) {
        throw ModuleException("Error parsing from C to Py, expected Int "+std::string(e.what()));
    }
}



/***Text parser ***/

TextParser::TextParser(const ColumnMeta& CM):UnitParser(CM){
    if (CM.size!=sizeof(char*))
        throw ModuleException("Bad size allocated for a text");
}

int16_t TextParser::py_to_c(PyObject* text, void* payload) const {
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

PyObject* TextParser::c_to_py(const void* payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to txtptr, found NULL");
    int64_t  *addr = (int64_t*) ((char*)payload);
    char *d = reinterpret_cast<char *>(*addr);
    if (d == nullptr) throw ModuleException("Error parsing from C to Py, expected ptr to text, found NULL");
    return PyUnicode_FromString(d);
}


/*** Numpy parser ***/

NumpyParser::NumpyParser(const ColumnMeta& CM):UnitParser(CM){
    if (CM.size!=sizeof(ArrayMetadata*))
        throw ModuleException("Bad size allocated for a Numpy");
    table=CM.info.at("table");
    attribute_name=CM.info.at("name");
    keyspace=CM.info.at("keyspace");
    //parse storage id
    uint64_t *uuid = (uint64_t*) CM.info.at("storage_id").c_str();
    storage_id = {*uuid,*(uuid+1)};
}

NumpyParser::~NumpyParser() {
    delete(np_storage);
}

int16_t NumpyParser::py_to_c(PyObject* numpy, void* payload) const {
    if (numpy==Py_None) return -1;
    PyArrayObject *arr;
    if (!PyArray_OutputConverter(numpy, &arr)) error_parsing("Numpy",numpy); //failed to convert array from PyObject to PyArray
    const ArrayMetadata *metas = np_storage->store(attribute_name,storage_id,arr);
    memcpy(payload,&metas,sizeof(ArrayMetadata*));
    return 0;
}

PyObject* NumpyParser::c_to_py(const void* payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to bytes, found NULL");
    //Receives a Arraymetadata
    const ArrayMetadata** metas = (const ArrayMetadata**) payload;
    PyObject *arr = np_storage->read(table,keyspace,attribute_name,storage_id,*metas);
    if (!arr) return Py_None;
    return  arr;
}
