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

/***
 * Returns the allocation number of bytes required to allocate a type of data
 * @param VT Cassandra Type
 * @return Allocation size needed to store data of type VT
 */
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
            //TODO
            std::cerr << "Parse decimals data type supported yet" << std::endl;
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
            //TODO
            std::cerr << "Timestamp data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_UUID: {
            //TODO

            std::cerr << "UUID data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_TIMEUUID: {
            std::cerr << "TIMEUUID data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_INET: {

            //TODO
            std::cerr << "INET data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_DATE: {
            std::cerr << "Date data type supported yet" << std::endl;

            //TODO
            break;
        }
        case CASS_VALUE_TYPE_TIME: {
            std::cerr << "Time data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_SMALL_INT: {
            return sizeof(int16_t);
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            return sizeof(int8_t);
        }
        case CASS_VALUE_TYPE_LIST: {
            std::cerr << "List data type supported yet" << std::endl;
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_MAP: {
            std::cerr << "Map data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_SET: {
            std::cerr << "Set data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_TUPLE: {
            //TODO

            std::cerr << "Tuple data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_UDT:
        case CASS_VALUE_TYPE_CUSTOM:
        case CASS_VALUE_TYPE_UNKNOWN:
        default:
            throw ModuleException("Unknown data type, can't parse");
            //TODO
    }
    return 0;
}



/*** TUPLE BUILDERS ***/

/***
 * Build a tuple taking the given buffer as its internal representation
 * @param data Valid pointer to the values
 * @return TupleRow with the buffer as its inner data
 * @post The TupleRow now owns the data and this cannot be freed
 */
TupleRow *TupleRowFactory::make_tuple(void * data) {
    return  new TupleRow(metadata, total_bytes, data);
}


/***
 * Build a tuple from the given Cassandra result row using the factory's metadata
 * @param row Contains the same number of columns than the metadata
 * @return TupleRow with a copy of the values inside the row
 * @post The row can be freed
 */
TupleRow *TupleRowFactory::make_tuple(const CassRow *row) {
    if (!row) return NULL;
    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, make tuple from CassRow: Null metadata");

    char *buffer = (char *) malloc(total_bytes);
    uint16_t i = 0;
    CassIterator *it = cass_iterator_from_row(row);
    while (cass_iterator_next(it)) {
        if (i>=localMeta->size())
            throw ModuleException("TupleRowFactory: Make tuple from CassRow: Access metadata at "+std::to_string(i)+" from a max "+std::to_string(localMeta->size()));
        cass_to_c(cass_iterator_get_column(it), buffer + localMeta->at(i).position, i);
        if (localMeta->at(i).position >= total_bytes)
            throw ModuleException("TupleRowFactory: Make tuple from CassRow: Writing on byte "+std::to_string(localMeta->at(i).position)+" from a total of "+std::to_string(total_bytes));
        if (i > localMeta->size())
            throw ModuleException("TupleRowFactory: Query has more columns than the ones retrieved from Cassandra");
        ++i;
    }
    cass_iterator_free(it);
    return new TupleRow(metadata, total_bytes, buffer);
}

/***
 * Build a tuple from the given Python object using the factory's metadata
 * @param obj Python List containing exactly the same number of objects that metadata size
 * @return TupleRow with a copy of the values in obj
 * @post The python object can be deleted
 */
TupleRow *TupleRowFactory::make_tuple(PyObject *obj) {
    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, make tuple from PyObj: Null metadata");

    char *buffer = (char *) malloc(total_bytes);
    for (uint16_t i = 0; i < PyList_Size(obj); ++i) {
        PyObject *obj_to_conver = PyList_GetItem(obj, i);
        if (i>=localMeta->size())
            throw ModuleException("TupleRowFactory: Make tuple from PyObj: Access metadata at "+std::to_string(i)+" from a max "+std::to_string(localMeta->size()));
        if (localMeta->at(i).position >= total_bytes)
            throw ModuleException("TupleRowFactory: Make tuple from PyObj: Writing on byte "+std::to_string(localMeta->at(i).position)+" from a total of "+std::to_string(total_bytes));
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

int TupleRowFactory::py_to_c(PyObject *obj, void *data, int32_t col) const {
    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, py_to_c: Null metadata");
    if (col < 0 || col >= (int32_t) localMeta->size()) {
        throw ModuleException("TupleRowFactory: Py to C: Asked for column "+std::to_string(col)+" but only "+std::to_string(localMeta->size())+" are present");
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
                throw ModuleException("TupleRowFactory: Py to c: Couldn't parse text");

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
            throw ModuleException("TupleRowFactory: Marshall from Py to C: Unsupported type not recognized by Cassandra");
    }
    return ok;
}


/***
 * @pre: -
 * @post: Extract the Cassandra's value from lhs and writes it to the memory pointed by data
 * using the data type information provided by type_array[col]
 * @param lhs Cassandra value
 * @param data Pointer to the place where the extracted value "lhs" should be written
 * @param col Indicates which column, thus, which data type are we processing
 * @return 0 if succeeds
 */
int TupleRowFactory::cass_to_c(const CassValue *lhs, void *data, int16_t col) const {
    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, cass_to_c: Null metadata");
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
            CassError rc = cass_value_get_string(lhs,&l_temp , &l_size);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse text unsuccessful, column"+std::to_string(col));
            char *permanent = (char*) malloc(l_size+1);
            memcpy(permanent, l_temp,l_size);
            permanent[l_size] = '\0';
            memcpy(data,&permanent,sizeof(char*));
            return 0;
        }
        case CASS_VALUE_TYPE_VARINT:
        case CASS_VALUE_TYPE_BIGINT: {
            int64_t *p = static_cast<int64_t * >(data);
            CassError rc = cass_value_get_int64(lhs, p);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse bigint/varint unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_BLOB: {
            const unsigned char *l_temp;
            size_t l_size;
            CassError rc = cass_value_get_bytes(lhs, &l_temp, &l_size);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse bytes unsuccessful, column:"+std::to_string(col));

            //Allolcate space for the bytes
            char *permanent = (char*) malloc(l_size+sizeof(uint32_t));
            //TODO make sure l_size < uint32 max
            uint32_t int_size = (uint32_t) l_size;

            //copy num bytes
            memcpy(permanent,&int_size,sizeof(uint32_t));

            //copy bytes
            memcpy(permanent+sizeof(uint32_t), l_temp,l_size);

            //copy pointer to payload
            memcpy(data,&permanent,sizeof(char*));
            return 0;
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
            cass_bool_t b;
            CassError rc = cass_value_get_bool(lhs, &b);
            CHECK_CASS("TupleRowFactory: Cassandra to C parse bool unsuccessful, column:"+std::to_string(col));
            bool *p = static_cast<bool *>(data);
            if (b == cass_true) *p = true;
            else *p = false;
            return 0;
        }
        case CASS_VALUE_TYPE_COUNTER: {
            CassError rc = cass_value_get_uint32(lhs, reinterpret_cast<uint32_t *>(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse counter as uint32 unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //TODO
            //decimal.Decimal
            break;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            CassError rc = cass_value_get_double(lhs, reinterpret_cast<double *>(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse double unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_FLOAT: {
            CassError rc = cass_value_get_float(lhs, reinterpret_cast<float * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse float unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_INT: {
            CassError rc = cass_value_get_int32(lhs, reinterpret_cast<int32_t * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse int32 unsuccessful, column:"+std::to_string(col));
            return 0;
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_UUID: {
            //TODO
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
            CassError rc = cass_value_get_int16(lhs, reinterpret_cast<int16_t * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse int16 unsuccessful, column:"+std::to_string(col));
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            CassError rc = cass_value_get_int8(lhs, reinterpret_cast<int8_t * >(data));
            CHECK_CASS("TupleRowFactory: Cassandra to C parse int16 unsuccessful, column:"+std::to_string(col));
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
            break;
    }
    return 0;
}


/***
 * @pre: Already owns the GIL lock, V points to C++ valid data
 * @param V is a pointer to the C++ valid value
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

PyObject *TupleRowFactory::tuple_as_py(const TupleRow *tuple) const {
    if (tuple == NULL) throw ModuleException("TupleRowFactory: Marshalling from c to python a NULL tuple, unsupported");
    PyObject *list = PyList_New(tuple->n_elem());
    const std::vector<ColumnMeta> *localMeta = metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, tuple_as_py: Null metadata");
    for (uint16_t i = 0; i < tuple->n_elem(); i++) {
        if (i >= localMeta->size())
            throw ModuleException("TupleRowFactory: Tuple as py access meta at " + std::to_string(i) + " from a max " +
                                  std::to_string(localMeta->size()));
        PyObject *inte = c_to_py(tuple->get_element(i), localMeta->at(i));
        PyList_SetItem(list, i, inte);
    }
    return list;
}
/***
 * Binds the tuple row data to the statement, using the offset to compute the position inside the query where the element needs to be bind
 * @param statement contains the query with elements to be binded
 * @param row data to use to bind in the statement
 * @param offset starting position inside the query from which the elements need to be bind
 */
void TupleRowFactory::bind( CassStatement *statement,const TupleRow *row,  u_int16_t offset) const  {

    const std::vector<ColumnMeta>* localMeta=metadata.get();
    if (!localMeta)
        throw ModuleException("Tuple row, tuple_as_py: Null metadata");

    for (uint16_t i = 0; i < row->n_elem(); ++i) {
        const void *key = row->get_element(i);

        uint32_t bind_pos = i + offset;
        if (i>=localMeta->size())
            throw ModuleException("TupleRowFactory: Binding element on pos "+std::to_string(bind_pos)+" from a max "+std::to_string(localMeta->size()));

        switch (localMeta->at(i).type) {
            case CASS_VALUE_TYPE_VARCHAR:
            case CASS_VALUE_TYPE_TEXT:
            case CASS_VALUE_TYPE_ASCII: {
                int64_t *addr = (int64_t *) key;
                const char *d = reinterpret_cast<char *>(*addr);
                CassError rc = cass_statement_bind_string(statement, bind_pos, d);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [text], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_VARINT:
            case CASS_VALUE_TYPE_BIGINT: {
                const int64_t *data = static_cast<const int64_t *>(key);
                CassError rc = cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [bigint/varint], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_BLOB: {
                int64_t  *addr = (int64_t*) key;
                const unsigned char *d = reinterpret_cast<char unsigned *>(*addr);
                uint32_t nbyes = *reinterpret_cast<uint32_t *>(*addr);
                CassError rc = cass_statement_bind_bytes(statement,bind_pos,d+sizeof(uint32_t),nbyes);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [bytes], column:"+localMeta->at(bind_pos).name);


                break;
            }
            case CASS_VALUE_TYPE_BOOLEAN: {
                cass_bool_t b = cass_false;
                const bool *bindbool = static_cast<const bool *>(key);
                if (*bindbool) b = cass_true;
                CassError rc = cass_statement_bind_bool(statement, bind_pos, b);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [bool], column:"+localMeta->at(bind_pos).name);
                break;
            }
            //TODO parsed as uint32 or uint64 on different methods
            case CASS_VALUE_TYPE_COUNTER: {
                const uint64_t *data = static_cast<const uint64_t *>(key);
                CassError rc = cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [counter as uint64], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_DECIMAL: {
                //decimal.Decimal
                //TODO
                break;
            }
            case CASS_VALUE_TYPE_DOUBLE: {
                const double *data = static_cast<const double *>(key);
                CassError rc = cass_statement_bind_double(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [double], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_FLOAT: {
                const float *data = static_cast<const float *>(key);
                CassError rc = cass_statement_bind_float(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [float], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_INT: {
                const int32_t *data = static_cast<const int32_t *>(key);
                CassError rc = cass_statement_bind_int32(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [int32], column:"+localMeta->at(bind_pos).name);
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
            case CASS_VALUE_TYPE_TIMEUUID: {
                //TODO
                break;
            }
            case CASS_VALUE_TYPE_INET: {
                //TODO
                break;
            }
            case CASS_VALUE_TYPE_DATE: {

                break;
            }
            case CASS_VALUE_TYPE_TIME: {
                //TODO
                break;
            }
            case CASS_VALUE_TYPE_SMALL_INT: {
                const int16_t *data = static_cast<const int16_t *>(key);
                CassError rc = cass_statement_bind_int16(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [small int as int16], column:"+localMeta->at(bind_pos).name);
                break;
            }
            case CASS_VALUE_TYPE_TINY_INT: {
                const int8_t *data = static_cast<const int8_t *>(key);
                CassError rc = cass_statement_bind_int8(statement, bind_pos, *data);
                CHECK_CASS("TupleRowFactory: Cassandra binding query unsuccessful [tiny int as int8], column:"+localMeta->at(bind_pos).name);
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
    }
}
