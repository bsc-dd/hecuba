#include "UnitParser.h"

int16_t UnitParser::py_to_c(PyObject *element, void *payload) const {
    throw ModuleException("Not implemented");
}

PyObject *UnitParser::c_to_py(const void *payload) const {
    throw ModuleException("Not implemented");
}


/*** Bool parser ***/

BoolParser::BoolParser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(bool))
        throw ModuleException("Bad size allocated for a Bool column");
}

int16_t BoolParser::py_to_c(PyObject *obj, void *payload) const {
    if (obj == Py_None) return -1;
    if (PyBool_Check(obj)) {
        bool *temp = static_cast<bool *>(payload);
        if (obj == Py_True) *temp = true;
        else *temp = false;
        return 0;
    }
    error_parsing("PyBool", obj);
    return -2;
}

PyObject *BoolParser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int, found NULL");
    const bool *temp = reinterpret_cast<const bool *>(payload);
    if (*temp) return Py_True;
    return Py_False;
}


/*** Int8 parser ***/

Int8Parser::Int8Parser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(int8_t))
        throw ModuleException("Bad size allocated for a Int8");
}

int16_t Int8Parser::py_to_c(PyObject *myint, void *payload) const {
    if (myint == Py_None) return -1;
    int8_t temp;
    if (PyInt_Check(myint) && PyArg_Parse(myint, Py_SHORT_INT, &temp)) {
        memcpy(payload, &temp, sizeof(int8_t));
        return 0;
    }
    error_parsing("PyInt as TinyInt", myint);
    return -2;
}

PyObject *Int8Parser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int8, found NULL");
    const int8_t *temp = reinterpret_cast<const int8_t *>(payload);
    return Py_BuildValue(Py_SHORT_INT, *temp);
}


/*** Int16 parser ***/

Int16Parser::Int16Parser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(int16_t))
        throw ModuleException("Bad size allocated for a Int16");
}

int16_t Int16Parser::py_to_c(PyObject *myint, void *payload) const {
    if (myint == Py_None) return -1;
    int16_t temp;
    if (PyInt_Check(myint) && PyArg_Parse(myint, Py_SHORT_INT, &temp)) {
        memcpy(payload, &temp, sizeof(int16_t));
        return 0;
    }
    error_parsing("PyInt as SmallInt", myint);
    return -2;
}

PyObject *Int16Parser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int16, found NULL");
    const int16_t *temp = reinterpret_cast<const int16_t *>(payload);
    return Py_BuildValue(Py_SHORT_INT, *temp);
}


/*** Int32 parser ***/

Int32Parser::Int32Parser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(int32_t))
        throw ModuleException("Bad size allocated for a Int32");
}

int16_t Int32Parser::py_to_c(PyObject *myint, void *payload) const {
    if (myint == Py_None) return -1;
    if (PyInt_Check(myint) && PyArg_Parse(myint, Py_INT, payload)) return 0;
    error_parsing("PyInt to Int32", myint);
    return -2;
}

PyObject *Int32Parser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int, found NULL");
    const int32_t *temp = reinterpret_cast<const int32_t *>(payload);
    return Py_BuildValue(Py_INT, *temp);
}


/*** Int64 parser ***/

Int64Parser::Int64Parser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(int64_t))
        throw ModuleException("Bad size allocated for a Int64");
}

int16_t Int64Parser::py_to_c(PyObject *myint, void *payload) const {
    if (myint == Py_None) return -1;
    if (PyInt_Check(myint) || PyLong_Check(myint)) {
        int64_t t; //TODO it might be safe to pass the payload instead of the var t
        if (PyArg_Parse(myint, Py_LONGLONG, &t) < 0) error_parsing("PyInt64", myint);
        memcpy(payload, &t, sizeof(t));
        return 0;
    }
    error_parsing("PyInt64", myint);
    return -2;
}

PyObject *Int64Parser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int64, found NULL");
    const int64_t *temp = reinterpret_cast<const int64_t *>(payload);
    return Py_BuildValue(Py_LONGLONG, *temp);
}

/*** Double parser ***/
/*** Called float in python ***/

DoubleParser::DoubleParser(const ColumnMeta &CM) : UnitParser(CM) {
    this->isFloat = false;
    if (CM.type == CASS_VALUE_TYPE_FLOAT) {
        this->isFloat = true;
        if (CM.size != sizeof(float))
            throw ModuleException("Bad size allocated for a PyDouble transformed to Float");
    } else if (CM.size != sizeof(double)) throw ModuleException("Bad size allocated for a PyDouble");
}

int16_t DoubleParser::py_to_c(PyObject *obj, void *payload) const {
    if (obj == Py_None) return -1;
    if (!PyFloat_Check(obj) && !PyInt_Check(obj)) error_parsing("PyDouble", obj);
    if (isFloat) {
        float t;
        if (!PyArg_Parse(obj, Py_FLOAT, &t)) error_parsing("PyDouble as Float", obj);
        memcpy(payload, &t, sizeof(t));
    } else {
        double t;
        if (!PyArg_Parse(obj, Py_DOUBLE, &t)) error_parsing("PyDouble as Double", obj);
        memcpy(payload, &t, sizeof(t));
    }
    return 0;
}

PyObject *DoubleParser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int, found NULL");
    if (isFloat) {
        const float *temp = reinterpret_cast<const float *>(payload);
        return Py_BuildValue(Py_FLOAT, *temp);
    } else {
        const double *temp = reinterpret_cast<const double *>(payload);
        return Py_BuildValue(Py_DOUBLE, *temp);
    }
}


/***Text parser ***/

TextParser::TextParser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(char *))
        throw ModuleException("Bad size allocated for a text");
}

int16_t TextParser::py_to_c(PyObject *text, void *payload) const {
    if (text == Py_None) return -1;
    if (PyString_Check(text) || PyUnicode_Check(text)) {
        char *l_temp;
        Py_ssize_t l_size;
        // PyString_AsStringAndSize returns internal string buffer of obj, not a copy.
        if (PyString_AsStringAndSize(text, &l_temp, &l_size) < 0) error_parsing("PyString", text);
        char *permanent = (char *) malloc(l_size + 1);
        memcpy(permanent, l_temp, l_size);
        permanent[l_size] = '\0';
        memcpy(payload, &permanent, sizeof(char *));
        return 0;
    }
    error_parsing("PyString", text);
    return -2;
}

PyObject *TextParser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to txtptr, found NULL");
    int64_t *addr = (int64_t *) ((char *) payload);
    char *d = reinterpret_cast<char *>(*addr);
    if (d == nullptr) throw ModuleException("Error parsing from C to Py, expected ptr to text, found NULL");
    return PyUnicode_FromString(d);
}


/***Bytes parser ***/

BytesParser::BytesParser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(char *))
        throw ModuleException("Bad size allocated for a text");
}

int16_t BytesParser::py_to_c(PyObject *obj, void *payload) const {
    if (obj == Py_None) return -1;
    if (PyByteArray_Check(obj)) {
        Py_ssize_t l_size = PyByteArray_Size(obj);
        char *l_temp = PyByteArray_AsString(obj);
        char *permanent = (char *) malloc(l_size + sizeof(uint64_t));
        uint64_t int_size = (uint64_t) l_size;
        if (int_size == 0) std::cerr << "array bytes has size 0" << std::endl; //Warning
        //copy num bytes
        memcpy(permanent, &int_size, sizeof(uint64_t));
        //copybytes
        memcpy(permanent + sizeof(uint64_t), l_temp, int_size);
        //copy pointer
        memcpy(payload, &permanent, sizeof(char *));
        return 0;
    }
    error_parsing("PyByteArray", obj);
    return -2;
}

PyObject *BytesParser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to txtptr, found NULL");
    int64_t *addr = (int64_t *) ((char *) payload);
    char *d = reinterpret_cast<char *>(*addr);
    if (d == nullptr) throw ModuleException("Error parsing from C to Py, expected ptr to text, found NULL");
    return PyUnicode_FromString(d);
}


/***UuidParser parser ***/

UuidParser::UuidParser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(uint64_t *))
        throw ModuleException("Bad size allocated for a text");
}

int16_t UuidParser::py_to_c(PyObject *obj, void *payload) const {
    if (obj == Py_None) return -1;
    if (!PyByteArray_Check(obj)) {
        //Object is UUID python class
        char *permanent = (char *) malloc(sizeof(uint64_t) * 2);

        memcpy(payload, &permanent, sizeof(char *));
        PyObject *bytes = PyObject_GetAttrString(obj, "time_low"); //32b
        if (!bytes)
            error_parsing("python UUID", obj);
        uint64_t time_low = (uint32_t) PyLong_AsLongLong(bytes);

        bytes = PyObject_GetAttrString(obj, "time_mid"); //16b
        uint64_t time_mid = (uint16_t) PyLong_AsLongLong(bytes);

        bytes = PyObject_GetAttrString(obj, "time_hi_version"); //16b
        uint64_t time_hi_version = (uint16_t) PyLong_AsLongLong(bytes);


        bytes = PyObject_GetAttrString(obj, "clock_seq_hi_variant"); //8b
        uint64_t clock_seq_hi_variant = (uint64_t) PyLong_AsLongLong(bytes);
        bytes = PyObject_GetAttrString(obj, "clock_seq_low"); //8b
        uint64_t clock_seq_low = (uint64_t) PyLong_AsLongLong(bytes);


        bytes = PyObject_GetAttrString(obj, "node"); //48b
        uint64_t second = (uint64_t) PyLong_AsLongLong(bytes);

        uint64_t first = (time_hi_version << 48) + (time_mid << 32) + (time_low);

        memcpy(permanent, &first, sizeof(first));
        permanent += sizeof(first);

        second += clock_seq_hi_variant << 56;
        second += clock_seq_low << 48;

        memcpy(permanent, &second, sizeof(second));
        return 0;
    } else throw ModuleException("Parsing UUID from ByteArray not supported");
}

PyObject *UuidParser::c_to_py(const void *payload) const {
    char **data = (char **) payload;
    char *it = *data;

    if (it == nullptr) throw ModuleException("Error parsing from C to Py, expected ptr to UUID bits, found NULL");
    char final[CASS_UUID_STRING_LENGTH];

    //trick to transform the data back, since it was parsed using the cassandra generator
    CassUuid uuid = {*((uint64_t *) it), *((uint64_t *) it + 1)};
    cass_uuid_string(uuid, final);
    return PyString_FromString(final);
}