#include "UnitParser.h"
#include <datetime.h>

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
    if (*temp) {
        Py_INCREF(Py_True);
        return Py_True;
    }
    Py_INCREF(Py_False);
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
    if (PyLong_Check(myint) && PyArg_Parse(myint, Py_SHORT_INT, &temp)) {
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


    if (PyLong_Check(myint) && PyArg_Parse(myint, Py_SHORT_INT, &temp)) {
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
    if (PyLong_Check(myint) && PyArg_Parse(myint, Py_INT, payload)) return 0;
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
    if (PyLong_Check(myint)) {
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
    if (!PyFloat_Check(obj) && !PyLong_Check(obj)) error_parsing("PyDouble", obj);
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
    if (PyUnicode_Check(text)) {
        Py_ssize_t l_size;
        const char *l_temp = PyUnicode_AsUTF8AndSize(text, &l_size);
        if (!l_temp) error_parsing("PyString", text);
        // l_temp points to the internal "text" memory buffer

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

/***Timestamp parser ***/

TimestampParser::TimestampParser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(int64_t *))
        throw ModuleException("Bad size allocated for a timestamp");
    if (!PyDateTimeAPI) PyDateTime_IMPORT;
}

int16_t TimestampParser::py_to_c(PyObject *obj, void *payload) const {
    if (obj == Py_None) return -1;
    if (PyDateTime_CheckExact(obj)) {
        time_t time_now;
        time(&time_now);
        struct tm timeinfo = {0}; //express datetime to the current timezone (tzset)
        timeinfo.tm_sec = PyDateTime_DATE_GET_SECOND(obj);
        timeinfo.tm_min = PyDateTime_DATE_GET_MINUTE(obj);
        timeinfo.tm_hour = PyDateTime_DATE_GET_HOUR(obj);
        timeinfo.tm_year = PyDateTime_GET_YEAR(obj) - 1900;
        timeinfo.tm_mon = PyDateTime_GET_MONTH(obj) - 1;
        timeinfo.tm_mday = PyDateTime_GET_DAY(obj);
        time_t time = mktime(&timeinfo);
        if(time == -1) throw ModuleException("Calendar time cannot be represented");
        auto diff = std::chrono::system_clock::from_time_t(time).time_since_epoch();
        std::time_t time_epoch = 0;
        time_t timezone = -1 * std::mktime(std::gmtime(&time_epoch));
        int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() + (timezone * 1000);
        memcpy(payload, &ms, sizeof(int64_t));
        return 0;
    }
    else { //if pyobject is a double it has already the exact date so is no use to call tzset
        if (!PyFloat_Check(obj) && !PyLong_Check(obj)) error_parsing("PyDouble", obj);
        double t;
        if (!PyArg_Parse(obj, Py_DOUBLE, &t)) error_parsing("PyDouble as Double", obj);
        time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::time_point(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::duration<double>(t))));
        auto diff = std::chrono::system_clock::from_time_t(time).time_since_epoch();
        std::time_t time_epoch = 0;
        time_t timezone = -1 * std::mktime(std::gmtime(&time_epoch));
        int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() + (timezone * 1000);
        memcpy(payload, &ms, sizeof(int64_t));
        return 0;
    }
    error_parsing("PyDateTime_DateType", obj);
    return -2;
}

PyObject *TimestampParser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int, found NULL");
    time_t time =  *(reinterpret_cast<const time_t *>(payload)) /1000; //we convert from ms to sec (UNIX time)
    struct tm * timeinfo = gmtime(&time); //gmt+0
    PyObject *timestamp_py = PyDateTime_FromDateAndTime(timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, 0);
    return timestamp_py;
}

/***Date parser ***/

DateParser::DateParser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(int64_t *))
        throw ModuleException("Bad size allocated for a date");
    if (!PyDateTimeAPI) PyDateTime_IMPORT;
}

int16_t DateParser::py_to_c(PyObject *obj, void *payload) const {
    if (obj == Py_None) return -1;
    if (PyDate_CheckExact(obj)) {
        struct tm timeinfo = {0}; //express datetime to the current timezone (tzset)
        timeinfo.tm_year = PyDateTime_GET_YEAR(obj) - 1900;
        timeinfo.tm_mon = PyDateTime_GET_MONTH(obj) - 1;
        timeinfo.tm_mday = PyDateTime_GET_DAY(obj);
        std::time_t time_epoch = 0;
        time_t time = mktime(&timeinfo);
        if(time == -1) throw ModuleException("Calendar time cannot be represented");
        time_t timezone = -1 * std::mktime(std::gmtime(&time_epoch));
        time += timezone;
        memcpy(payload, &time, sizeof(uint32_t *));
        return 0;
    }
    error_parsing("PyDateTime_DateType", obj);
    return -2;
}

PyObject *DateParser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int, found NULL");
    const time_t *time = reinterpret_cast<const time_t *>(payload);
    std::tm *now = std::gmtime(time);
    PyObject *date_py = PyDate_FromDate(now->tm_year + 1900, now->tm_mon + 1, now->tm_mday);
    return date_py;
}

/***Time parser ***/



TimeParser::TimeParser(const ColumnMeta &CM) : UnitParser(CM) {
    if (CM.size != sizeof(int64_t *))
        throw ModuleException("Bad size allocated for a time");
    if (!PyDateTimeAPI) PyDateTime_IMPORT;
}

int16_t TimeParser::py_to_c(PyObject *obj, void *payload) const {
    if (obj == Py_None) return -1;
    if (PyTime_CheckExact(obj)) {
        int64_t date = static_cast<int64_t>(PyDateTime_TIME_GET_HOUR(obj)) * 3600000000000 + //time in nanoseconds
                       static_cast<int64_t>(PyDateTime_TIME_GET_MINUTE(obj)) * 60000000000 +
                       static_cast<int64_t>(PyDateTime_TIME_GET_SECOND(obj)) * 1000000000 +
                       PyDateTime_TIME_GET_MICROSECOND(obj) * 1000;
        memcpy(payload, &date, sizeof(int64_t));
        return 0;
    }
    error_parsing("PyDateTime_DateType", obj);
    return -2;
}

PyObject *TimeParser::c_to_py(const void *payload) const {
    if (!payload) throw ModuleException("Error parsing from C to Py, expected ptr to int, found NULL");
    int64_t msec = *(reinterpret_cast<const int64_t *>(payload)) / 1000; //from nanoseconds to microseconds
    int64_t hour = 0, min = 0, sec = 0;
    hour = msec / 3600000000;
    msec = msec - 3600000000 * hour;
    //60000000 microseconds in a minute
    min = msec / 60000000;
    msec = msec - 60000000 * min;
    //1000000 microseconds in a second
    sec = msec / 1000000;
    msec = msec - 1000000 * sec;
    PyObject *time_py = PyTime_FromTime(hour, min, sec, msec);
    return time_py;
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
    this->uuid_module = PyImport_ImportModule("uuid");
    if (!this->uuid_module) throw ModuleException("Error importing the module 'uuid'");
    Py_INCREF(uuid_module);
}

UuidParser::~UuidParser() {
    Py_DECREF(this->uuid_module);
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
        bytes = PyObject_GetAttrString(obj, "bytes"); //64b
        if (!bytes)
            error_parsing("python UUID bytes", obj);
        char *uuid = PyBytes_AsString(bytes);
        if (!uuid)
            error_parsing("python UUID  2 bytes ", obj);
        memcpy(permanent, uuid, 16); // Keep the UUID as is (RFC4122)

        return 0;
    } else throw ModuleException("Parsing UUID from ByteArray not supported");
}

PyObject *UuidParser::c_to_py(const void *payload) const {
    char **data = (char **) payload;
    char *it = *data;

    if (it == nullptr) throw ModuleException("Error parsing from C to Py, expected ptr to UUID bits, found NULL");
    char final[CASS_UUID_STRING_LENGTH];

#if 1
    //trick to transform the data back, since it was parsed using the cassandra generator
    CassUuid uuid = {*((uint64_t *) it), *((uint64_t *) it + 1)};

    // CassUuid has a different format than UUID RFC4122
    // 'data' has been saved in RFC4122, and here we read that value, transform
    // it to CassUUID, generate the "standard string" from it, and call
    // uuid.UUID(string) to reconstruct the python uuid object... it should be
    // easier just to call uuid.UUID(bytes=it) if we know how to do it.

    CassUuid tmp_uuid;
    char *p = (char*)&(tmp_uuid.time_and_version);
    char *psrc = (char*)&uuid.time_and_version;
    // Recode time_low
    p[0] = psrc[3];
    p[1] = psrc[2];
    p[2] = psrc[1];
    p[3] = psrc[0];

    // Recode time_mid
    p[4] = psrc[5];
    p[5] = psrc[4];

    // Recode time_hi_&_version
    p[6] = psrc[7];
    p[7] = psrc[6];

    // Recode clock_seq_and_node
    p= (char*)&(tmp_uuid.clock_seq_and_node);
    psrc = (char*)&uuid.clock_seq_and_node;

    for (uint32_t ix=0; ix<8;ix++)
        p[ix] = psrc[7-ix];

    cass_uuid_string(tmp_uuid, final);
    PyObject *uuidpy = PyObject_CallMethod(this->uuid_module, "UUID", "s", final);
#else
    PyObject *uuidpy = PyObject_CallMethod(this->uuid_module, "UUID", "sss(s)L", NULL, NULL, NULL, NULL, it);
#endif
    if (!uuidpy) throw ModuleException("Error parsing UUID from C to Py, expected a non-NULL result");
    return uuidpy;
}

TupleParser::TupleParser(const ColumnMeta &CM) : UnitParser(CM) {
    this->col_meta = CM;
}


int16_t TupleParser::py_to_c(PyObject *obj, void *payload) const {
    if (obj == Py_None) throw ModuleException("Error parsing PyObject from py to c, expected a non-none object");
    if (!PyTuple_Check(obj)) throw ModuleException("Error parsing PyObject from py to c, expected a tuple object");
    size_t size = col_meta.pointer->size();
    if ((size_t)PyTuple_Size(obj) != size)
        throw ModuleException(
                "Error parsing PyObject from py to c, expected size of Py_tuple being the same as Column_meta");
    uint32_t total_malloc = 0;
    for (uint32_t i = 0; i < size; ++i) {
        total_malloc = total_malloc + col_meta.pointer->at(i).size;
    }
    void *internal_payload = malloc(total_malloc);
    TupleRow *tr = new TupleRow(col_meta.pointer, total_malloc, internal_payload);
    memcpy(payload, &tr, sizeof(tr));

    for (uint32_t i = 0; i < size; ++i) {
        PyObject *tuple_elem = PyTuple_GetItem(obj, i);
        CassValueType cvt = this->col_meta.pointer->at(i).type;
        void *destiny = (char *) internal_payload + this->col_meta.pointer->at(i).position;
        if (tuple_elem != Py_None) {
            switch (cvt) {
                case CASS_VALUE_TYPE_VARCHAR:
                case CASS_VALUE_TYPE_TEXT:
                case CASS_VALUE_TYPE_ASCII: {
                    TextParser tp = TextParser(col_meta.pointer->at(i));
                    tp.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_VARINT:
                case CASS_VALUE_TYPE_BIGINT: {
                    Int64Parser i64p = Int64Parser(col_meta.pointer->at(i));
                    i64p.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_BLOB: {
                    BytesParser bp = BytesParser(col_meta.pointer->at(i));
                    bp.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_BOOLEAN: {
                    BoolParser bp = BoolParser(col_meta.pointer->at(i));
                    bp.py_to_c(tuple_elem, destiny);
                    break;
                }
                    //TODO parsed as uint32 or uint64 on different methods
                case CASS_VALUE_TYPE_COUNTER: {
                    Int64Parser i64p = Int64Parser(col_meta.pointer->at(i));
                    i64p.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_DECIMAL: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_DOUBLE: {

                }
                case CASS_VALUE_TYPE_FLOAT: {
                    DoubleParser dp = DoubleParser(col_meta.pointer->at(i));
                    dp.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_INT: {
                    Int32Parser i32p = Int32Parser(col_meta.pointer->at(i));
                    i32p.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_TIMESTAMP: {
                    TimestampParser dp = TimestampParser(col_meta.pointer->at(i));
                    dp.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_UUID: {
                    UuidParser uip = UuidParser(col_meta.pointer->at(i));
                    uip.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_TIMEUUID: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_INET: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_DATE: {
                    DateParser dp = DateParser(col_meta.pointer->at(i));
                    dp.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_TIME: {
                    TimeParser dp = TimeParser(col_meta.pointer->at(i));
                    dp.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_SMALL_INT: {
                    Int16Parser i16p = Int16Parser(col_meta.pointer->at(i));
                    i16p.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_TINY_INT: {
                    Int8Parser i8p = Int8Parser(col_meta.pointer->at(i));
                    i8p.py_to_c(tuple_elem, destiny);
                    break;
                }
                case CASS_VALUE_TYPE_LIST: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_MAP: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_SET: {
                    //TODO
                    break;
                }
                default:
                    break;
            }
        } else {
            tr->setNull(i);
        }

    }
    return 0;
}


PyObject *TupleParser::c_to_py(const void *payload) const {

    if (payload == nullptr) throw ModuleException("Error parsing payload from c to py, expected a non-null payload");

    TupleRow **ptr = (TupleRow **) payload;
    const TupleRow *inner_data = *ptr;

    size_t size = col_meta.pointer->size();
    PyObject *tuple = PyTuple_New(size);
    for (uint32_t i = 0; i < size; ++i) {
        CassValueType cvt = this->col_meta.pointer->at(i).type;
        if (!inner_data->isNull(i)) {
            switch (cvt) {
                case CASS_VALUE_TYPE_VARCHAR:
                case CASS_VALUE_TYPE_TEXT:
                case CASS_VALUE_TYPE_ASCII: {
                    TextParser tp = TextParser(col_meta.pointer->at(i));
                    int64_t *p = (int64_t *) inner_data->get_element(i);
                    PyObject *po = tp.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_VARINT:
                case CASS_VALUE_TYPE_BIGINT: {
                    Int64Parser i64p = Int64Parser(col_meta.pointer->at(i));
                    int64_t *p = (int64_t *) inner_data->get_element(i);
                    PyObject *po = i64p.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_BLOB: {
                    BytesParser bp = BytesParser(col_meta.pointer->at(i));
                    int64_t *p = (int64_t *) inner_data->get_element(i);
                    PyObject *po = bp.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_BOOLEAN: {
                    BoolParser bp = BoolParser(col_meta.pointer->at(i));
                    double_t *p = (double_t *) inner_data->get_element(i);
                    PyObject *po = bp.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                    //TODO parsed as uint32 or uint64 on different methods
                case CASS_VALUE_TYPE_COUNTER: {
                    Int64Parser i64p = Int64Parser(col_meta.pointer->at(i));
                    int64_t *p = (int64_t *) inner_data->get_element(i);
                    PyObject *po = i64p.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_DECIMAL: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_DOUBLE: {
                    throw ModuleException("Float type not supported");
                }
                case CASS_VALUE_TYPE_FLOAT: {
                    DoubleParser dp = DoubleParser(col_meta.pointer->at(i));
                    double_t *p = (double_t *) inner_data->get_element(i);
                    PyObject *po = dp.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;

                }
                case CASS_VALUE_TYPE_INT: {
                    Int32Parser i32p = Int32Parser(col_meta.pointer->at(i));
                    int32_t *p = (int32_t *) inner_data->get_element(i);
                    PyObject *po = i32p.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_TIMESTAMP: {
                    TimestampParser uip = TimestampParser((col_meta.pointer->at(i)));
                    int64_t *p = (int64_t *) inner_data->get_element(i);
                    PyObject *po = uip.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_UUID: {
                    UuidParser uip = UuidParser((col_meta.pointer->at(i)));
                    uint64_t *p = (uint64_t *) inner_data->get_element(i);
                    PyObject *po = uip.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_TIMEUUID: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_INET: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_DATE: {
                    DateParser uip = DateParser((col_meta.pointer->at(i)));
                    int64_t *p = (int64_t *) inner_data->get_element(i);
                    PyObject *po = uip.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_TIME: {
                    TimeParser uip = TimeParser((col_meta.pointer->at(i)));
                    int64_t *p = (int64_t *) inner_data->get_element(i);
                    PyObject *po = uip.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_SMALL_INT: {
                    Int16Parser i16p = Int16Parser(col_meta.pointer->at(i));
                    int16_t *p = (int16_t *) inner_data->get_element(i);
                    PyObject *po = i16p.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_TINY_INT: {
                    Int8Parser i8p = Int8Parser(col_meta.pointer->at(i));
                    int8_t *p = (int8_t *) inner_data->get_element(i);
                    PyObject *po = i8p.c_to_py(p);
                    PyTuple_SET_ITEM(tuple, i, po);
                    break;
                }
                case CASS_VALUE_TYPE_LIST: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_MAP: {
                    //TODO
                    break;
                }
                case CASS_VALUE_TYPE_SET: {
                    //TODO
                    break;
                }
                default:
                    break;
            }
        } else {
            Py_INCREF(Py_None);
            PyTuple_SET_ITEM(tuple, i, Py_None);
        }
    }
    return tuple;
}
