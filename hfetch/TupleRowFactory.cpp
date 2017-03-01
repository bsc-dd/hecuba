#include "TupleRowFactory.h"

/***
 * Builds a tuple factory to retrieve tuples based on rows and keys
 * extracting the information from Cassandra to decide the types to be used
 * @param table_meta Holds the table information
 */
TupleRowFactory::TupleRowFactory(const CassTableMeta *table_meta, const std::vector< std::vector<std::string> > &col_names) {

    if (!table_meta) {
        throw ModuleException("Tuple factory: Table metadata NULL");
    }
    uint32_t ncols = (uint32_t) col_names.size();
    if (ncols==0) {
        throw ModuleException("Tuple factory: 0 columns metadata");
    }

    //SELECT ALL
    if (col_names[0][0] == "*") ncols = (uint32_t) cass_table_meta_column_count(table_meta);


    std::vector<ColumnMeta> md = std::vector< ColumnMeta>(ncols);
    std::vector<uint16_t> elem_sizes = std::vector<uint16_t>(ncols);

    CassIterator *iterator = cass_iterator_columns_from_table_meta(table_meta);
    while (cass_iterator_next(iterator)) {
        const CassColumnMeta *cmeta = cass_iterator_get_column_meta(iterator);
        const char *value;
        size_t length;
        cass_column_meta_name(cmeta, &value, &length);
        const CassDataType *type = cass_column_meta_data_type(cmeta);

        std::string meta_col_name(value);


        for (uint16_t j=0; j<col_names.size(); ++j) {
            const std::string ss = col_names[j][0];
            if (meta_col_name == ss) {
                md[j] ={0,cass_data_type_type(type),col_names[j]};
                elem_sizes[j] = compute_size_of( md[j].type);
                break;
            }
        }

    }
    cass_iterator_free(iterator);

    md[0].position = 0;
    uint16_t i = 1;
    while (i < ncols) {
        md[i].position= md[i - 1].position + elem_sizes[i - 1];
        ++i;
    }

    this->total_bytes = md[md.size()-1].position + elem_sizes[ncols - 1];
    this->metadata=std::make_shared<std::vector<ColumnMeta>>(md);
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
    auto localMeta=metadata.get();
    while (cass_iterator_next(it)) {
        cass_to_c(cass_iterator_get_column(it), buffer + localMeta->at(i).position, i);
        if (i > localMeta->size())
            throw ModuleException("TupleRowFactory: Query has more columns than the ones retrieved from Cassandra");
        ++i;
    }
    cass_iterator_free(it);
    TupleRow *t = new TupleRow(metadata, total_bytes, buffer);
    return t;
}


TupleRow *TupleRowFactory::make_tuple(PyObject *obj) {
    char *buffer = (char *) malloc(total_bytes);
    auto localMeta=metadata.get();

    for (uint16_t i = 0; i < PyList_Size(obj); ++i) {
        PyObject *obj_to_conver = PyList_GetItem(obj, i);
        py_to_c(obj_to_conver, buffer + localMeta->at(i).position, i);
    }

    TupleRow *t = new TupleRow(metadata, total_bytes, buffer);
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
    auto localMeta=this->metadata.get();
    if (col < 0 || col >= (int32_t) localMeta->size()) {
        throw ModuleException("TupleRowFactory: Py to C: Asked for column "+std::to_string(col)+" but only "+std::to_string(localMeta->size())+" are present");
    }
    if (key == Py_None) {
        memset(data,0,compute_size_of(localMeta->at(col).type));
        return 0;
    }
    int ok = -1;
    switch (localMeta->at(col).type) {
        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_ASCII: {
            char *l_temp;
            Py_ssize_t l_size;
            PyString_AsStringAndSize(key,&l_temp,&l_size);

            char *permanent = (char*) malloc(l_size+1);
            memcpy(permanent, l_temp,l_size);
            permanent[l_size] = '\0';
            memcpy(data,&permanent,sizeof(char*));
            break;
        }
        case CASS_VALUE_TYPE_BIGINT: {
            ok = PyArg_Parse(key, "L", data);
            break;
        }
        case CASS_VALUE_TYPE_BLOB: {
            /** interface receives key **/

           // _import_array();
            PyArrayObject *arr;
            ok =PyArray_OutputConverter(key,&arr);
            if (!ok) throw ModuleException("error parsing PyArray to obj");
            /** transform to bytes **/
            PyObject* bytes = PyArray_ToString(arr, NPY_KEEPORDER);
            /** encode as Hex **/
            PyObject *encoded= PyString_AsEncodedObject(bytes,"hex",NULL);


            Py_ssize_t l_size;
            char *l_temp;
            ok = PyString_AsStringAndSize(encoded,&l_temp,&l_size);
            char *permanent = (char*) malloc(l_size+1);
            memcpy(permanent, l_temp,l_size);
            permanent[l_size] = '\0';
            memcpy(data,&permanent,sizeof(char*));
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
            cass_float_t t;
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
    auto localMeta=this->metadata.get();
    if (col < 0 || col >= (int32_t) localMeta->size()) {
        throw ModuleException("TupleRowFactory: Cass to C: Asked for column "+std::to_string(col)+" but only "+std::to_string(localMeta->size())+" are present");
    }

    if (cass_value_is_null(lhs)) {
        memset(data, 0, compute_size_of(localMeta->at(col).type));
        return 0;
    }

    switch (localMeta->at(col).type) {
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




            const cass_byte_t *l_temp;
            size_t l_size;
            cass_value_get_bytes(lhs,&l_temp , &l_size);
            char *permanent = (char*) malloc(l_size);

            memcpy(permanent, l_temp,l_size);
            permanent[l_size]='\0';

            memcpy(data,&permanent,sizeof(char*));
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

PyObject *TupleRowFactory::tuple_as_py(const TupleRow *tuple) const {
    if (tuple == 0) throw ModuleException("TupleRowFactory: Marshalling from c to python a NULL tuple, unsupported");
    PyObject *list = PyList_New(tuple->n_elem());
    auto localMeta=metadata.get();
    for (uint16_t i = 0; i < tuple->n_elem(); i++) {
        PyObject *inte=c_to_py(tuple->get_element(i), localMeta->at(i));
        PyList_SetItem(list, i,inte);
    }
    return list;


}

/***
 * PRE: Already owns the GIL lock, V points to C++ valid data
 * @param V Pointer to the C++ valid value
 * @param VT Data type in Cassandra Types
 * @return The equivalent object V in Python using the Cassandra Value type to choose the correct transformation
 */

PyObject *TupleRowFactory::c_to_py(const void *V, ColumnMeta &meta) const {

    char const *py_flag = 0;
    PyObject *py_value = Py_None;
    if (V == 0) {
        return py_value;
    }

    switch (meta.type) {
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_ASCII: {
            int64_t  *addr = (int64_t*) V;
            char *d = reinterpret_cast<char *>(*addr);
            py_value = PyUnicode_FromString(d);
            break;
        }
        case CASS_VALUE_TYPE_BIGINT: {
            py_flag = Py_LONGLONG;
            const int64_t *temp = reinterpret_cast<const int64_t *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
            break;
        }
        case CASS_VALUE_TYPE_BLOB: {//bytes

            int64_t  *addr = (int64_t*) V;
            char *d = reinterpret_cast<char *>(*addr);

            _import_array(); //necessary only for running tests
            PyObject*bytes=PyString_FromString(d);
            PyObject* decoded=PyString_AsDecodedObject(bytes,"hex",NULL);
            PyErr_Print();

            char *dec = PyString_AsString(decoded);
            py_value = PyArray_FromString(dec,PyString_GET_SIZE(decoded),PyArray_DescrNewFromType(meta.get_arr_type()),-1,NULL);

            PyArrayObject *arr;
            int ok =PyArray_OutputConverter(py_value,&arr);
            if (ok) py_value = PyArray_Reshape (arr, meta.get_arr_dims());
            break;
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
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


void TupleRowFactory::bind( CassStatement *statement,const TupleRow *row,  u_int16_t offset) const  {
    for (uint16_t i = 0; i < row->n_elem(); ++i) {

        const void *key = row->get_element(i);
        uint16_t bind_pos = i + offset;
        switch (get_type(i)) {
            case CASS_VALUE_TYPE_VARCHAR:
            case CASS_VALUE_TYPE_TEXT:
            case CASS_VALUE_TYPE_ASCII: {
                int64_t *addr = (int64_t *) key;
                const char *d = reinterpret_cast<char *>(*addr);
                cass_statement_bind_string(statement, bind_pos, d);
                break;
            }
            case CASS_VALUE_TYPE_VARINT:
            case CASS_VALUE_TYPE_BIGINT: {
                const int64_t *data = static_cast<const int64_t *>(key);
                cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                break;
            }
            case CASS_VALUE_TYPE_BLOB: {
                int64_t  *addr = (int64_t*) key;
                const unsigned char *d = reinterpret_cast<char unsigned *>(*addr);
                cass_statement_bind_bytes(statement,bind_pos,d,strlen(reinterpret_cast<char *>(*addr)));
                break;
            }
            case CASS_VALUE_TYPE_BOOLEAN: {
                cass_bool_t b = cass_false;
                const bool *bindbool = static_cast<const bool *>(key);
                if (*bindbool) b = cass_true;
                cass_statement_bind_bool(statement, bind_pos, b);
                break;
            }
            case CASS_VALUE_TYPE_COUNTER: {
                const uint64_t *data = static_cast<const uint64_t *>(key);
                cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                break;
            }
            case CASS_VALUE_TYPE_DECIMAL: {
                //decimal.Decimal
                break;
            }
            case CASS_VALUE_TYPE_DOUBLE: {
                const double *data = static_cast<const double *>(key);
                cass_statement_bind_double(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_FLOAT: {
                const float *data = static_cast<const float *>(key);
                cass_statement_bind_float(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_INT: {
                const int32_t *data = static_cast<const int32_t *>(key);
                cass_statement_bind_int32(statement, bind_pos, *data);
                break;
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
                const int16_t *data = static_cast<const int16_t *>(key);
                cass_statement_bind_int16(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_TINY_INT: {
                const int8_t *data = static_cast<const int8_t *>(key);
                cass_statement_bind_int8(statement, bind_pos, *data);
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
    }
}
