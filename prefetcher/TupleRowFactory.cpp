//
// Created by bscuser on 1/26/17.
//

#include "TupleRowFactory.h"

/***
 * Builds a tuple factory to retrieve tuples based on rows and keys
 * extracting the information from Cassandra to decide the types to be used
 * @param table_meta Holds the table information
 */
TupleRowFactory::TupleRowFactory(const CassTableMeta *table_meta, const std::vector<std::string> &col_names) {

    if (!table_meta) {
        throw ModuleException("Tuple factory: Table metadata NULL");
    }

    CassIterator *iterator = cass_iterator_columns_from_table_meta(table_meta);

    uint32_t ncols = (uint32_t) col_names.size();
    if (col_names[0] == "*") ncols = (uint32_t) cass_table_meta_column_count(table_meta);

    if (ncols==0) {
        throw ModuleException("Tuple factory: 0 columns metadata");
    }


    type_array = std::vector<CassValueType>(ncols);
    name_map = std::vector<std::string>(ncols);

    this->offsets = std::vector<uint16_t>(ncols);

    std::vector<uint16_t> elem_sizes = std::vector<uint16_t>(ncols);

    uint16_t i = 0;

    while (cass_iterator_next(iterator)) {
        const CassColumnMeta *cmeta = cass_iterator_get_column_meta(iterator);
        const char *value;
        size_t length;
        cass_column_meta_name(cmeta, &value, &length);
        const CassDataType *type = cass_column_meta_data_type(cmeta);

        std::string meta_col_name(value);
        for (const std::string &ss: col_names) {
            if (meta_col_name == ss) {
                type_array[i] = cass_data_type_type(type);
                name_map[i] = value;
                elem_sizes[i] = compute_size_of(type_array[i]);
                ++i;
                //break;
            }
        }

    }
    cass_iterator_free(iterator);

    offsets[0] = 0;
    i = 1;
    while (i < ncols) {
        offsets[i] = offsets[i - 1] + elem_sizes[i - 1];
        ++i;
    }

    this->total_bytes = offsets[ncols - 1] + elem_sizes[ncols - 1];
}


uint16_t TupleRowFactory::compute_size_of(const CassValueType VT) const {
    switch (VT) {

        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_ASCII: {
            return sizeof(char *);
        }
        case CASS_VALUE_TYPE_VARINT:
        case CASS_VALUE_TYPE_BIGINT: {
            return sizeof(int64_t);
        }
        case CASS_VALUE_TYPE_BLOB: {
            return sizeof(unsigned char *);
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
            return sizeof(bool);
        }
        case CASS_VALUE_TYPE_COUNTER: {
            return sizeof(uint32_t);
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //decimal.Decimal
            break;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            return sizeof(double);
        }
        case CASS_VALUE_TYPE_FLOAT: {
            return sizeof(float);
        }
        case CASS_VALUE_TYPE_INT: {
            return sizeof(int32_t);
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {
            break;
        }
        case CASS_VALUE_TYPE_UUID: {

            break;
        }
        case CASS_VALUE_TYPE_TIMEUUID: {

            break;
        }
        case CASS_VALUE_TYPE_INET: {

            break;
        }
        case CASS_VALUE_TYPE_DATE: {

            break;
        }
        case CASS_VALUE_TYPE_TIME: {

            break;
        }
        case CASS_VALUE_TYPE_SMALL_INT: {
            return sizeof(int16_t);
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            return sizeof(int8_t);
        }
        case CASS_VALUE_TYPE_LIST: {
            break;
        }
        case CASS_VALUE_TYPE_MAP: {

            break;
        }
        case CASS_VALUE_TYPE_SET: {

            break;
        }
        case CASS_VALUE_TYPE_TUPLE: {

            break;
        }
        case CASS_VALUE_TYPE_UDT:
        case CASS_VALUE_TYPE_CUSTOM:
        case CASS_VALUE_TYPE_UNKNOWN:
        default://
            break;
    }
    return 0;
}


TupleRow *TupleRowFactory::make_tuple(const CassRow *row) {
    if (!row) return 0;

    char *buffer = (char *) malloc(total_bytes);
    uint16_t i = 0;
    CassIterator *it = cass_iterator_from_row(row);
    while (cass_iterator_next(it)) {
        cass_to_c(cass_iterator_get_column(it), buffer + offsets[i], i);
        if (i > offsets.size())
            throw ModuleException("TupleRowFactory: Query has more columns than the ones retrieved from Cassandra");
        ++i;
    }
    cass_iterator_free(it);
    TupleRow *t = new TupleRow(&offsets, total_bytes, buffer);
    return t;
}


TupleRow *TupleRowFactory::make_tuple(PyObject *obj) {
    char *buffer = (char *) malloc(total_bytes);

    for (uint16_t i = 0; i < PyList_Size(obj); ++i) {
        PyObject *obj_to_conver = PyList_GetItem(obj, i);
        py_to_c(obj_to_conver, buffer + offsets[i], i);
    }

    TupleRow *t = new TupleRow(&offsets, total_bytes, buffer);
    return t;
}

/***
 * PRE: Already owns the GIL lock
 * POST: Writes on the memory pointed by data the python object key parse using the type specified by type_array[col]
 * @param key Python object to be parsed
 * @param data Place to write the data
 * @param col Indicates which column, thus, which data type are we processing
 * @return 0 if succeeds
 */

int TupleRowFactory::py_to_c(PyObject *key, void *data, int32_t col) const {
    if (col < 0 || col >= (int32_t) type_array.size()) {
        throw ModuleException("TupleRowFactory: Asked for column "+std::to_string(col)+" but only "+std::to_string(type_array.size())+" are present");
    }
    if (key == Py_None) return 0;
    int ok = -1;
    switch (type_array[col]) {
        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_ASCII: {
            const char *text;
            ok = PyArg_Parse(key, "s", text);
            memcpy(data, &text, sizeof(text));
            break;
        }
        case CASS_VALUE_TYPE_BIGINT: {
            ok = PyArg_Parse(key, "L", data);
            break;
        }
        case CASS_VALUE_TYPE_BLOB: {
            PyObject *size = PyInt_FromSsize_t(PyByteArray_Size(key));
            int32_t c_size;
            PyArg_Parse(size, Py_INT, &c_size);;
            const char *bytes = PyByteArray_AsString(key);
            memcpy(data, &bytes, sizeof(bytes));
            break;
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
            bool *temp = static_cast<bool *>(data);
            if (key == Py_True) *temp = true;
            else *temp = false;
            break;
        }
        case CASS_VALUE_TYPE_COUNTER: {
            ok = PyArg_Parse(key, Py_U_LONGLONG, data);
            break;
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //decimal.Decimal
            return 0;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            cass_double_t t;
            ok = PyArg_Parse(key, Py_DOUBLE, &t);
            memcpy(data, &t, sizeof(t));

            break;
        }
        case CASS_VALUE_TYPE_FLOAT: {
            cass_float_t t = 0.003;
            ok = PyArg_Parse(key, Py_FLOAT, &t); /* A string */
            memcpy(data, &t, sizeof(t));
            break;
        }
        case CASS_VALUE_TYPE_INT: {
            int32_t t;
            ok = PyArg_Parse(key, Py_INT, &t); /* A string */
            memcpy(data, &t, sizeof(t));
            break;
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {

            return 0;
        }
        case CASS_VALUE_TYPE_UUID: {

            return 0;
        }
        case CASS_VALUE_TYPE_VARINT: {
            ok = PyArg_Parse(key, Py_LONG, data);
            break;
        }
        case CASS_VALUE_TYPE_TIMEUUID: {

            break;
        }
        case CASS_VALUE_TYPE_INET: {

            break;
        }
        case CASS_VALUE_TYPE_DATE: {

            break;
        }
        case CASS_VALUE_TYPE_TIME: {

            break;
        }
        case CASS_VALUE_TYPE_SMALL_INT: {
            ok = PyArg_Parse(key, Py_INT, data);
            break;
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            ok = PyArg_Parse(key, Py_SHORT_INT, data);
            break;
        }
        case CASS_VALUE_TYPE_LIST: {
            break;
        }
        case CASS_VALUE_TYPE_MAP: {

            break;
        }
        case CASS_VALUE_TYPE_SET: {

            break;
        }
        case CASS_VALUE_TYPE_TUPLE: {

            break;
        }
        case CASS_VALUE_TYPE_UDT:
        case CASS_VALUE_TYPE_CUSTOM:
        case CASS_VALUE_TYPE_UNKNOWN:
            break;
        default:
            throw ModuleException("TupleRowFactory: Marshall from Py to C: Unsupported type not recognized by Cassandra");
    }
    return ok;
}


/***
 * PRE: -
 * POST: Extract the Cassandra value from lhs and writes it to the memory pointed by data
 * using the data type information provided by type_array[col]
 * @param lhs Cassandra value
 * @param data Pointer to the place where the extracted value "lhs" should be written
 * @param col Indicates which column, thus, which data type are we processing
 * @return 0 if succeeds
 */
int TupleRowFactory::cass_to_c(const CassValue *lhs, void *data, int16_t col) const {

    if (col < 0 || col >= (int32_t) type_array.size()) {
        throw ModuleException("TupleRowFactory: Asked for column "+std::to_string(col)+" but only "+std::to_string(type_array.size())+" are present");
    }

    if (cass_value_is_null(lhs)) {
        memset(data, 0, compute_size_of(type_array[col]));
        return 0;
    }

    switch (type_array[col]) {
        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_ASCII: {
            const char *l_temp;
            size_t l_size;
            cass_value_get_string(lhs,&l_temp , &l_size);
            char *permanent = (char*) malloc(l_size+1);
            memcpy(permanent, l_temp,l_size);
            permanent[l_size] = '\0';
            memcpy(data,&permanent,sizeof(char*));
            return 0;
        }
        case CASS_VALUE_TYPE_VARINT:
        case CASS_VALUE_TYPE_BIGINT: {
            int64_t *p = static_cast<int64_t * >(data);
            cass_value_get_int64(lhs, p);
            return 0;
        }
        case CASS_VALUE_TYPE_BLOB: {
            const unsigned char *l_temp;
            size_t l_size;
            cass_value_get_bytes(lhs, &l_temp, &l_size);
            memcpy(data, &l_temp, sizeof(l_temp));
            return 0;
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
            cass_bool_t b;
            cass_value_get_bool(lhs, &b);
            bool *p = static_cast<bool *>(data);
            if (b == cass_true) *p = true;
            else *p = false;
            return 0;
        }
        case CASS_VALUE_TYPE_COUNTER: {
            cass_value_get_uint32(lhs, reinterpret_cast<uint32_t *>(data));
            return 0;
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //decimal.Decimal
            break;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            cass_value_get_double(lhs, reinterpret_cast<double *>(data));
            return 0;
        }
        case CASS_VALUE_TYPE_FLOAT: {
            cass_value_get_float(lhs, reinterpret_cast<float * >(data));
            return 0;
        }
        case CASS_VALUE_TYPE_INT: {
            cass_value_get_int32(lhs, reinterpret_cast<int32_t * >(data));
            return 0;
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {

            break;
        }
        case CASS_VALUE_TYPE_UUID: {

            break;
        }
        case CASS_VALUE_TYPE_TIMEUUID: {

            break;
        }
        case CASS_VALUE_TYPE_INET: {

            break;
        }
        case CASS_VALUE_TYPE_DATE: {

            break;
        }
        case CASS_VALUE_TYPE_TIME: {

            break;
        }
        case CASS_VALUE_TYPE_SMALL_INT: {
            cass_value_get_int16(lhs, reinterpret_cast<int16_t * >(data));
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            cass_value_get_int8(lhs, reinterpret_cast<int8_t * >(data));
        }
        case CASS_VALUE_TYPE_LIST: {
            break;
        }
        case CASS_VALUE_TYPE_MAP: {

            break;
        }
        case CASS_VALUE_TYPE_SET: {

            break;
        }
        case CASS_VALUE_TYPE_TUPLE: {

            break;
        }
        default://CASS_VALUE_TYPE_UDT|CASS_VALUE_TYPE_CUSTOM|CASS_VALUE_TYPE_UNKNOWN:
            break;
    }
    return 0;
}

/***
 * Builds a Python list from the data being held inside the TupleRow
 * @param tuple
 * @return A list with the information from tuple preserving its order
 */

PyObject *TupleRowFactory::tuple_as_py(TupleRow *tuple) const {
    if (tuple == 0) throw ModuleException("TupleRowFactory: Marshalling from c to python a NULL tuple, unsupported");
    PyObject *list = PyList_New(0);
    for (uint16_t i = 0; i < tuple->n_elem(); i++) {
        PyList_Append(list, c_to_py(tuple->get_element(i), type_array[i]));
    }
    return list;


}

/***
 * PRE: Already owns the GIL lock, V points to C++ valid data
 * @param V Pointer to the C++ valid value
 * @param VT Data type in Cassandra Types
 * @return The equivalent object V in Python using the Cassandra Value type to choose the correct transformation
 */

PyObject *TupleRowFactory::c_to_py(const void *V, CassValueType VT) const {

    char const *py_flag = 0;
    PyObject *py_value = Py_None;
    if (V == 0) {
        return py_value;
    }

    switch (VT) {
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_ASCII: {
            uint64_t addr;
            memcpy(&addr,V,sizeof(char*));
            char *d = reinterpret_cast<char *>(addr);
            py_value = PyUnicode_FromString(d);
            //py_value = Py_BuildValue(py_flag, d);
            break;
        }
        case CASS_VALUE_TYPE_BIGINT: {
            py_flag = Py_LONGLONG;
            const int64_t *temp = reinterpret_cast<const int64_t *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
            break;
        }
        case CASS_VALUE_TYPE_BLOB: {//bytes
            break;
        }
        case CASS_VALUE_TYPE_BOOLEAN: {//bool
            break;
        }
        case CASS_VALUE_TYPE_COUNTER: {
            py_flag = Py_U_LONGLONG;
            const int64_t *temp = reinterpret_cast<const int64_t *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
            break;
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //decimal.Decimal
            break;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            py_flag = Py_DOUBLE;
            const double *temp = reinterpret_cast<const double *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
            break;
        }
        case CASS_VALUE_TYPE_FLOAT: {
            py_flag = Py_FLOAT;
            const float *temp = reinterpret_cast<const float *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
            break;
        }
        case CASS_VALUE_TYPE_INT: {
            py_flag = Py_INT;
            const int32_t *temp = reinterpret_cast<const int32_t *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
            break;
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {

            break;
        }
        case CASS_VALUE_TYPE_UUID: {

            break;
        }
        case CASS_VALUE_TYPE_VARINT: {
            py_flag = Py_LONG;
            const int64_t *temp = reinterpret_cast<const int64_t *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
            break;
        }
        case CASS_VALUE_TYPE_TIMEUUID: {

            break;
        }
        case CASS_VALUE_TYPE_INET: {

            break;
        }
        case CASS_VALUE_TYPE_DATE: {

            break;
        }
        case CASS_VALUE_TYPE_TIME: {

            break;
        }

        case CASS_VALUE_TYPE_SMALL_INT: {

            py_flag = Py_INT;
            const int16_t *temp = reinterpret_cast<const int16_t *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
            break;
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            py_flag = Py_SHORT_INT;
            const int8_t *temp = reinterpret_cast<const int8_t *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
            break;
        }
        case CASS_VALUE_TYPE_LIST: {
            break;
        }
        case CASS_VALUE_TYPE_MAP: {

            break;
        }
        case CASS_VALUE_TYPE_SET: {

            break;
        }
        case CASS_VALUE_TYPE_TUPLE: {

            break;
        }
        default://CASS_VALUE_TYPE_UDT|CASS_VALUE_TYPE_CUSTOM|CASS_VALUE_TYPE_UNKNOWN:
            break;
    }
    return py_value;
}
