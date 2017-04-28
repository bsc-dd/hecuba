#include "PythonParser.h"

/***
 * Builds a tuple factory to retrieve tuples based on rows and keys
 * extracting the information from Cassandra to decide the types to be used
 * @param table_meta Holds the table information
 */
PythonParser::PythonParser() {

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
    uint32_t total_bytes = localMeta->at(localMeta->size()-1).position+localMeta->at(localMeta->size()-1).size;

    char *buffer = (char *) malloc(total_bytes);
    for (uint16_t i = 0; i < PyList_Size(obj); ++i) {
        PyObject *obj_to_conver = PyList_GetItem(obj, i);
        if (i>=localMeta->size())
            throw ModuleException("PythonParser: Make tuple from PyObj: Access metadata at "+std::to_string(i)+" from a max "+std::to_string(localMeta->size()));
        if (localMeta->at(i).position >= total_bytes)
            throw ModuleException("PythonParser: Make tuple from PyObj: Writing on byte "+std::to_string(localMeta->at(i).position)+" from a total of "+std::to_string(total_bytes));
        py_to_c(obj_to_conver, buffer + localMeta->at(i).position, localMeta->at(i).type);
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

int PythonParser::py_to_c(PyObject *obj, void *data, CassValueType type) const {
    int ok = -1;
    switch (type) {
        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_ASCII: {
            char *l_temp;
            Py_ssize_t l_size;
            ok = PyString_AsStringAndSize(obj, &l_temp, &l_size);
            if (ok < 0)
                throw ModuleException("TupleRowFactory: Py to c: Couldn't parse text");
            char *permanent = (char *) malloc(l_size + 1);
            memcpy(permanent, l_temp, l_size);
            permanent[l_size] = '\0';
            memcpy(data, &permanent, sizeof(char *));
            break;
        }
        case CASS_VALUE_TYPE_BIGINT: {
            ok = PyArg_Parse(obj, "L", data);
            break;
        }
        case CASS_VALUE_TYPE_BLOB: {
            /** interface receives key **/
            //TODO test
            Py_ssize_t l_size = PyByteArray_Size(obj);
            char *l_temp = PyByteArray_AsString(obj);
            char *permanent = (char *) malloc(l_size + sizeof(uint64_t));
            uint64_t int_size = (uint64_t) l_size;
            if (int_size == 0) std::cerr << "array bytes has size 0" << std::endl;
            //copy num bytes
            memcpy(permanent, &int_size, sizeof(uint64_t));
            //copybytes
            memcpy(permanent + sizeof(uint64_t), l_temp, int_size);
            //copy pointer
            memcpy(data, &permanent, sizeof(char *));
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
            if (!PyByteArray_Check(obj)) {

                uint32_t len = sizeof(uint64_t)*2;

                char *permanent = (char *) malloc(len);

                memcpy(data, &permanent, sizeof(char *));
                PyObject *bytes = PyObject_GetAttrString(obj, "time_low"); //32b
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

                uint64_t first = (time_hi_version<< 48) + (time_mid << 32) + (time_low );

                memcpy(permanent, &first, sizeof(first));
                permanent += sizeof(first);

                second += clock_seq_hi_variant << 56;
                second += clock_seq_low << 48;;

                memcpy(permanent, &second, sizeof(second));

            } else {
                uint32_t len = sizeof(uint64_t)*2;

                char *permanent = (char *) malloc(len);

                memcpy(data, &permanent, sizeof(char *));
                char *cpp_bytes = NULL;
                cpp_bytes = PyByteArray_AsString(obj);
                if (!cpp_bytes)
                    throw ModuleException("Bytes null");

                memcpy(permanent, cpp_bytes, len);
            }
            break;
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
            //TODO
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

PyObject *PythonParser::c_to_py(const void *V,const ColumnMeta &meta) const {

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
            int64_t *addr = (int64_t *) V;
            char *d = reinterpret_cast<char *>(*addr);
            //d points to [uint32,bytearray] which stands for num_bytes and bytes

            uint64_t nbytes = *reinterpret_cast<uint64_t * >(d);
            d += sizeof(uint64_t);

            PyErr_Clear();
            try {
                //_import_array(); //necessary only for running tests
                py_value = PyArray_FromString(d, nbytes, PyArray_DescrNewFromType(get_arr_type(meta)), -1, NULL);
                PyArrayObject *arr;
                int ok = PyArray_OutputConverter(py_value, &arr);
                if (!ok) throw ModuleException("TupleRowFactory failed to convert array from PyObject to PyArray");
                PyArray_Dims *dims = get_arr_dims(meta);
                py_value = PyArray_Newshape(arr, dims, NPY_CORDER);

            }
            catch (std::exception e) {
                if (PyErr_Occurred()) PyErr_Print();
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return NULL;
            }
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
            PyErr_Print();
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
            char **data = (char **) V;

            char *it = *data;
            char final[CASS_UUID_STRING_LENGTH];

            CassUuid uuid = {*((uint64_t *) it), *((uint64_t *) it + 1)};
            cass_uuid_string(uuid, final);
            py_value = PyString_FromString(final);
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

PyObject *PythonParser::tuples_as_py(std::vector<const TupleRow *> &values, std::shared_ptr<const std::vector<ColumnMeta> > metadata) const {

    PyObject *list;
        //store to cache
        const TupleRow *tuple = values[0];
        if (tuple == 0)
            throw ModuleException("TupleRowFactory: Marshalling from c to python a NULL tuple, unsupported");
        list = PyList_New(tuple->n_elem());
        for (uint16_t i = 0; i < tuple->n_elem(); i++) {
            if (i >= metadata->size())
                throw ModuleException(
                        "TupleRowFactory: Tuple as py access meta at " + std::to_string(i) + " from a max " +
                        std::to_string(metadata->size()));
            PyObject *inte = c_to_py(tuple->get_element(i), metadata->at(i));
            PyList_SetItem(list, i, inte);
        }
    return list;

}


PyObject *PythonParser::merge_blocks_as_nparray(std::vector<const TupleRow *> &blocks,std::shared_ptr<const std::vector<ColumnMeta> > metadata) const {
    //for each member in metadata:
    // if member is simple: c_to_py
    // elif numpy:
    //  iterate over all blocks, get bytes and concatenate
    //  build an array and add to list;
    //  skip next metadata

    PyObject *list = PyList_New(0);
    if (blocks.empty()) throw ModuleException("No blocks on merge nparray");

    for (uint16_t pos = 0; pos < metadata->size(); ++pos) {
        //Object is not an array, process as usual
        if (metadata->at(pos).info.size() == 1 || get_arr_type(metadata->at(pos)) == NPY_NOTYPE) {
            PyObject *inte = c_to_py(blocks[0]->get_element(pos), metadata->at(pos));

           int ok = PyList_Append(list, inte);
            Py_DECREF(inte);
            if (ok < 0)
                throw ModuleException(
                        "Can't append object in position" + std::to_string(pos) + " toPyList on merge np array");
        } else if (metadata->at(pos).info.size() == 5) {
            //external, Add none, because it will get replaced
            PyObject* data_null = PyInt_FromLong(0);
             PyList_Append(list, data_null);
            Py_DECREF(data_null);
        } else {
            //object is a numpy array
            uint64_t nbytes = 0;
            for (const TupleRow *block:blocks) {
                void **data = (void **) block->get_element(pos);
                uint64_t *block_bytes = (uint64_t *) *data;
                nbytes += *block_bytes;
            }

            char *final_array = (char *) malloc(nbytes);
            if (metadata->at(pos).info.size() < 2) throw ModuleException("Info size is less than 2");
            if (metadata->at(pos).info[3] == "partition") {
                for (uint32_t i = 0; i < blocks.size(); ++i) {
                    char **data = (char **) blocks[i]->get_element(pos);
                    uint64_t *block_bytes = (uint64_t *) *data;

                    uint32_t *block_id = (uint32_t *) blocks[i]->get_element(pos + (uint16_t) 1);

                    memcpy(final_array + *block_id * maxarray_size, *data + sizeof(uint64_t), *block_bytes);
                }
            } else {
                char **data = (char **) blocks[0]->get_element(pos);
                uint64_t *block_bytes = (uint64_t *) *data;
                char *bytes_array = *data + sizeof(uint64_t);

                memcpy(final_array, bytes_array, *block_bytes);
            }
            //build np array from final_array data
            PyErr_Clear();
            try {
                PyObject *py_value = PyArray_SimpleNewFromData(get_arr_dims(metadata->at(pos))->len,
                                                               get_arr_dims(metadata->at(pos))->ptr,
                                                               get_arr_type(metadata->at(pos)), final_array);

                int ok = PyList_Append(list, py_value);
                Py_DECREF(py_value);
                if (ok < 0) throw ModuleException("Can't append numpy array into the results list");
            }
            catch (std::exception e) {
                if (PyErr_Occurred()) PyErr_Print();
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return NULL;
            }
            ++pos;
        }
    }
    return list;

}


const NPY_TYPES PythonParser::get_arr_type(const ColumnMeta& column_meta) const {
    if (column_meta.info.size()  < 2) {
        return NPY_NOTYPE;
    }
    if (column_meta.info[1] == "bool")
        return NPY_BOOL;
    if (column_meta.info[1] == "byte")
        return NPY_BYTE;
    if (column_meta.info[1] == "ubyte")
        return NPY_UBYTE;
    if (column_meta.info[1] == "short")
        return NPY_SHORT;
    if (column_meta.info[1] == "ushort")
        return NPY_USHORT;
    if (column_meta.info[1] == "int")
        return NPY_INT;
    if (column_meta.info[1] == "uint")
        return NPY_UINT;
    if (column_meta.info[1] == "long")
        return NPY_LONG;
    if (column_meta.info[1] == "ulong")
        return NPY_ULONG;
    if (column_meta.info[1] == "longlong")
        return NPY_LONGLONG;
    if (column_meta.info[1] == "float")
        return NPY_FLOAT;
    if (column_meta.info[1] == "double")
        return NPY_DOUBLE;
    if (column_meta.info[1] == "clongdouble")
        return NPY_LONGDOUBLE;
    if (column_meta.info[1] == "cfloat")
        return NPY_CFLOAT;
    if (column_meta.info[1] == "cdouble")
        return NPY_CDOUBLE;
    if (column_meta.info[1] == "clongdouble")
        return NPY_CLONGDOUBLE;
    if (column_meta.info[1] == "obj")
        return NPY_OBJECT;
    if (column_meta.info[1] == "str")
        return NPY_STRING;
    if (column_meta.info[1] == "unicode")
        return NPY_UNICODE;
    if (column_meta.info[1] == "void")
        return NPY_VOID;
    return NPY_NOTYPE;
}

PyArray_Dims *PythonParser::get_arr_dims(const ColumnMeta& column_meta) const {
    if (column_meta.info.size() < 3)
        throw ModuleException("Numpy array metadata must consist of [name,type,dimensions,partition]");

    std::string temp = column_meta.info[2];

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




//for each np array transform into tuples, if in the next positions arrays are to be found, ignore them
std::vector<const TupleRow *> PythonParser::make_tuples_with_npy(PyObject *obj, std::shared_ptr<const std::vector<ColumnMeta> > metadata) {
    uint16_t total_bytes = metadata->at(metadata->size()-1).position+metadata->at(metadata->size()-1).size;
    uint16_t nelem = (uint16_t) PyList_Size(obj);
    if (nelem > metadata->size())
        throw ModuleException(
                "TupleRowFactory: Make tuple from NUMPY: Access metadata at " + std::to_string(nelem - 1));
    if (metadata->at(nelem - (uint16_t) 1).position >= total_bytes)
        throw ModuleException("TupleRowFactory: Make tuple from NUMPY: Writing on byte " +
                              std::to_string(metadata->at(nelem - (uint16_t) 1).position));

//find arrays
    std::vector<const TupleRow *> tuples = std::vector<const TupleRow *>();
    for (uint16_t i = 0; i < PyList_Size(obj); ++i) {
        if (get_arr_type(metadata->at(i)) != NPY_NOTYPE && metadata->at(i).info[3] == "partition") {
            //found a np array
            std::vector<void *> blocks;
            blocks = split_array(PyList_GetItem(obj, i));
            std::vector<const TupleRow *> partitioned = blocks_to_tuple(blocks, metadata, obj);
            tuples.insert(tuples.begin(), partitioned.begin(), partitioned.end());
        } else if (get_arr_type(metadata->at(i)) != NPY_NOTYPE) {
            void *block = extract_array(PyList_GetItem(obj, i));
            //build the tuple with the first block and other information
            char *buffer = (char *) malloc(total_bytes);
            for (uint16_t j = 0; j < PyList_Size(obj); ++j) {
                PyObject *obj_to_conver = PyList_GetItem(obj, j);
                if (i != j) {
                    py_to_c(obj_to_conver, buffer + metadata->at(j).position, metadata->at(j).type);
                } else {
                    //numpy
                    memcpy(buffer + metadata->at(j).position, &block, sizeof(void *));
                }
            }
            tuples.push_back(new TupleRow(metadata, total_bytes, buffer));
        }
    }
    return tuples;
}




std::vector<const TupleRow *> PythonParser::blocks_to_tuple(std::vector<void *> &blocks, std::shared_ptr<const std::vector<ColumnMeta> > metadata, PyObject *obj) const {
    uint32_t total_bytes = metadata->at(metadata->size()-1).position+metadata->at(metadata->size()-1).size;

/*** FIRST BLOCK ***/
    //build the first tuple with the first block and other information
    char *buffer = (char *) malloc(total_bytes);
    for (uint16_t i = 0; i < PyList_Size(obj); ++i) {
        PyObject *obj_to_conver = PyList_GetItem(obj, i);
        //copy
        if (get_arr_type(metadata->at(i)) == NPY_NOTYPE) {
            //average column
            py_to_c(obj_to_conver, buffer + metadata->at(i).position, metadata->at(i).type);
        } else {
            //numpy
            memcpy(buffer + metadata->at(i).position, &blocks[0], sizeof(void *));
            //block position column
            ++i; //skip metadata column
            uint32_t block_id = 0;
            if (metadata->at(i).type != CASS_VALUE_TYPE_INT)
                throw ModuleException("Expected uint32 when building the first subarray of numpy, found cass type: " +
                                      std::to_string(metadata->at(i).type));
            //copy
            memcpy(buffer + metadata->at(i).position, &block_id, sizeof(uint32_t));
        }
    }


    std::vector<const TupleRow *> tuple_blocks(blocks.size());
    //first tuple holds all data
    tuple_blocks[0] = new TupleRow(metadata, total_bytes, buffer);

    //build the rest
    for (uint32_t nb = 1; nb < blocks.size(); ++nb) {
        buffer = (char *) malloc(total_bytes);
        for (uint16_t i = 0; i < metadata->size(); ++i) {
            if (get_arr_type(metadata->at(i)) != NPY_NOTYPE) {
                //copy bytes
                memcpy(buffer + metadata->at(i).position, &blocks[nb], sizeof(void *));

                //copy block position
                memcpy(buffer + metadata->at(++i).position, &nb, sizeof(uint32_t));
            } else {
                //its safer to set to 0, so when they are written into cassandra no exceptions should occur
                memset(buffer + metadata->at(i).position, 0, sizeof(void *));
            }
        }
        tuple_blocks[nb] = new TupleRow(metadata, total_bytes, buffer);
    }
    return tuple_blocks;
}



void *PythonParser::extract_array(PyObject *py_array) const {
    ssize_t nbytes_s = 0;
    void *original_payload = NULL;
    int ok;
    PyArrayObject *arr;
    try {
      //  _import_array();
        ok = PyArray_OutputConverter(py_array, &arr);
        if (!ok) throw ModuleException("error parsing PyArray to obj");
        original_payload = PyArray_DATA(arr);
        nbytes_s = PyArray_NBYTES(arr);
    }
    catch (std::exception e) {
        if (PyErr_Occurred()) PyErr_Print();
        std::cerr << e.what() << std::endl;
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    //then we split the payload
    uint64_t block_size = (uint64_t) nbytes_s;

    void *data_copy = malloc(block_size + sizeof(uint64_t));
    char *block = (char *) data_copy;

    //copy number of bytes
    memcpy(block, &block_size, sizeof(uint64_t));
    block += sizeof(uint64_t);

    //copy bytes
    memcpy(block, ((char *) original_payload), block_size);
    return data_copy;
}



std::vector<void *> PythonParser::split_array(PyObject *py_array) {
    //we have an array so we extract the bytes
    uint64_t nbytes = 0;
    void *data = NULL;

    PyArrayObject *arr;
    try {
      //  _import_array();
        if (!PyArray_OutputConverter(py_array, &arr))
            throw ModuleException("error parsing PyArray to obj");
        data = PyArray_DATA(arr);
        ssize_t nbytes_s = PyArray_NBYTES(arr);
        if (nbytes_s < 0)
            throw ModuleException("PyArray returns negative size of array");
        nbytes = (uint64_t) nbytes_s;
    }
    catch (std::exception e) {
        throw ModuleException(e.what());
    }

    //then we split the payload
    uint32_t nblocks = (uint32_t) (nbytes - (nbytes % maxarray_size))/maxarray_size; //number of subarrays
    if (nbytes % maxarray_size != 0) ++nblocks; //we don't want to lose data
    uint64_t block_size = std::min(nbytes, (uint64_t) maxarray_size);//bytes per block

    std::vector<void *> blocks_list(nblocks);


    for (uint32_t block_id = 0; block_id < nblocks; ++block_id) {
        if (block_id == nblocks - 1) block_size = nbytes - ((nblocks - 1) * block_size);
        //copy address of block
        blocks_list[block_id] = malloc(block_size + sizeof(uint64_t));
        char *block = (char *) blocks_list[block_id];

        memcpy(blocks_list[block_id], &block, sizeof(void *));

        //copy number of bytes
        memcpy(block, &block_size, sizeof(uint64_t));
        block += sizeof(uint64_t);

        //copy bytes
        memcpy(block, ((char *) data) + block_id * block_size, block_size);
    }

    return blocks_list;
}