#include "PythonParser.h"

/***
 * Builds a tuple factory to retrieve tuples based on rows and keys
 * extracting the information from Cassandra to decide the types to be used
 * @param table_meta Holds the table information
 */
PythonParser::PythonParser() {

    if (!table_meta) {
        throw ModuleException("Tuple factory: Table metadata NULL");
    }

    uint32_t ncols = (uint32_t) col_names.size();
    if (col_names[0] == "*") ncols = (uint32_t) cass_table_meta_column_count(table_meta);


    if (ncols==0) {
        throw ModuleException("Tuple factory: 0 columns metadata");
    }

    const std::vector<ColumnMeta> md = std::vector<ColumnMeta>(ncols);
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
            const std::string ss = col_names[j];
            if (meta_col_name == ss || col_names[0] == "*") {
                md[j] ={0,cass_data_type_type(type),value};
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
    this->metadata=std::make_shared<const std::vector<ColumnMeta>>(md);
}

PythonParser::~PythonParser() {

}
/*** TUPLE BUILDERS ***/

/***
 * Build a tuple from the given Python object using the factory's metadata
 * @param obj Python List containing exactly the same number of objects that metadata size
 * @return TupleRow with a copy of the values in obj
 * @post The python object can be deleted
 */
TupleRow *PythonParser::make_tuple(PyObject* obj,std::shared_ptr<const std::vector<ColumnMeta> > metadata) const {
    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, make tuple from PyObj: Null metadata");

    char *buffer = (char *) malloc(total_bytes);
    for (uint16_t i = 0; i < PyList_Size(obj); ++i) {
        PyObject *obj_to_conver = PyList_GetItem(obj, i);
        if (i>=localMeta->size())
            throw ModuleException("PythonParser: Make tuple from PyObj: Access metadata at "+std::to_string(i)+" from a max "+std::to_string(localMeta->size()));
        if (localMeta->at(i).position >= total_bytes)
            throw ModuleException("PythonParser: Make tuple from PyObj: Writing on byte "+std::to_string(localMeta->at(i).position)+" from a total of "+std::to_string(total_bytes));
        py_to_c(obj_to_conver, buffer + localMeta->at(i).position, i);
    }
    return new TupleRow(metadata, total_bytes, buffer);
}

/***
 * @pre: Already owns the GIL lock
 * @post: Writes the python object data into the memory pointed by data, parsing it accordingly to the type specified by type_array[col]
 * @param key Python object to be parsed
 * @param data Place to write the data
 * @param col Indicates which column, thus, which data type are we processing
 * @return 0 if succeeds, -1 otherwise
 */

int PythonParser::py_to_c(PyObject *obj, void *data, int32_t col) const {
    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, py_to_c: Null metadata");
    if (col < 0 || col >= (int32_t) localMeta->size()) {
        throw ModuleException("PythonParser: Py to C: Asked for column "+std::to_string(col)+" but only "+std::to_string(localMeta->size())+" are present");
    }
    if (obj == Py_None) {
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
            ok = PyString_AsStringAndSize(obj,&l_temp,&l_size);
            if (ok<0)
                throw ModuleException("PythonParser: Py to c: Couldn't parse text");

            if (l_size<0)
                throw ModuleException("Parsed string from python to c has length < 0");

            //Allocate space for the string
            char *permanent = (char*) malloc((size_t)l_size+1);
            memcpy(permanent, l_temp,(size_t)l_size);
            //Add termination flag
            permanent[l_size] = '\0';
            //Copy the address of the string to the data
            memcpy(data,&permanent,sizeof(char*));
            break;
        }
        case CASS_VALUE_TYPE_BIGINT: {
            ok = PyArg_Parse(obj, "L", data);
            break;
        }
        case CASS_VALUE_TYPE_BLOB: {
            //Parse python bytearray
            Py_ssize_t l_size = PyString_Size(obj);
            char *l_temp = PyByteArray_AsString(obj);
            if (l_size<0)
                throw ModuleException("Parsed string from python to c has length < 0");
            //Allocate space for the bytes
            char *permanent = (char*) malloc(l_size+sizeof(uint32_t));
            uint32_t int_size =(uint32_t) l_size;
            //copy the number of bytes
            memcpy(permanent,&int_size,sizeof(uint32_t));
            //copy the bytes contiguously
            memcpy(permanent+sizeof(uint32_t), l_temp,int_size);
            //copy the pointer to the bytes
            memcpy(data,&permanent,sizeof(char*));
            break;
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
            bool *temp = static_cast<bool *>(data);
            if (obj == Py_True) *temp = true;
            else *temp = false;
            break;
        }
        case CASS_VALUE_TYPE_COUNTER: {
            ok = PyArg_Parse(obj, Py_U_LONGLONG, data);
            break;
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //TODO
            //decimal.Decimal
            return 0;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            cass_double_t t;
            ok = PyArg_Parse(obj, Py_DOUBLE, &t);
            memcpy(data, &t, sizeof(t));
            break;
        }
        case CASS_VALUE_TYPE_FLOAT: {
            cass_float_t t;
            ok = PyArg_Parse(obj, Py_FLOAT, &t); /* A string */
            memcpy(data, &t, sizeof(t));
            break;
        }
        case CASS_VALUE_TYPE_INT: {
            int32_t t;
            ok = PyArg_Parse(obj, Py_INT, &t); /* A string */
            memcpy(data, &t, sizeof(t));
            break;
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {
            //TODO
            return 0;
        }
        case CASS_VALUE_TYPE_UUID: {
            //TODO
            return 0;
        }
        case CASS_VALUE_TYPE_VARINT: {
            ok = PyArg_Parse(obj, Py_LONG, data);
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
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_TIME: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_SMALL_INT: {
            ok = PyArg_Parse(obj, Py_INT, data);
            break;
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            ok = PyArg_Parse(obj, Py_SHORT_INT, data);
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
        case CASS_VALUE_TYPE_TUPLE: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_UDT:
        case CASS_VALUE_TYPE_CUSTOM:
        case CASS_VALUE_TYPE_UNKNOWN:
        default:
            throw ModuleException("PythonParser: Marshall from Py to C: Unsupported type not recognized by Cassandra");
    }
    return ok;
}

/***
 * @pre: Already owns the GIL lock, V points to C++ valid data
 * @param V is a pointer to the C++ valid value
 * @param VT Data type in Cassandra Types
 * @return The equivalent object V in Python using the Cassandra Value type to choose the correct transformation
 */

PyObject *PythonParser::c_to_py(const void *V, ColumnMeta &meta) const {

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
        case CASS_VALUE_TYPE_BLOB: {
            int64_t  *addr = (int64_t*) V;
            char *d = reinterpret_cast<char *>(*addr);
            //d points to [uint32,bytearray] which stands for num_bytes and bytes
            uint32_t nbytes = *reinterpret_cast<uint32_t* >(d);
            d+=sizeof(uint32_t);
            //TODO build a bytearray from bytes pointed by d, numbytes is nbytes
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
            //TODO
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
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_UUID: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_VARINT: {
            py_flag = Py_LONG;
            const int64_t *temp = reinterpret_cast<const int64_t *>(V);
            py_value = Py_BuildValue(py_flag, *temp);
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
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_TIME: {
            //TODO
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
        case CASS_VALUE_TYPE_TUPLE: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_UDT:
        case CASS_VALUE_TYPE_CUSTOM:
        case CASS_VALUE_TYPE_UNKNOWN:
        default:
            //TODO
            break;
    }
    return py_value;
}

/***
 * Builds a Python list from the data being held inside the TupleRow
 * @param tuple
 * @return A list with the information from tuple preserving its order
 */

PyObject *PythonParser::tuple_as_py(const TupleRow* tuple, std::shared_ptr<const std::vector<ColumnMeta> > metadata) const {
    if (tuple == NULL) throw ModuleException("PythonParser: Marshalling from c to python a NULL tuple, unsupported");
    PyObject *list = PyList_New(tuple->n_elem());
    const std::vector<ColumnMeta> *localMeta = metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, tuple_as_py: Null metadata");
    for (uint16_t i = 0; i < tuple->n_elem(); i++) {
        if (i >= localMeta->size())
            throw ModuleException("PythonParser: Tuple as py access meta at " + std::to_string(i) + " from a max " +
                                  std::to_string(localMeta->size()));
        PyObject *inte = c_to_py(tuple->get_element(i), localMeta->at(i));
        PyList_SetItem(list, i, inte);
    }
    return list;
}


