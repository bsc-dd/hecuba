#include "PythonParser.h"


PythonParser::PythonParser(std::shared_ptr<StorageInterface> storage, std::shared_ptr<const std::vector<ColumnMeta> > metadatas) {
    this->metas = metadatas;
    this->parsers = std::vector<UnitParser*>(metadatas->size());
    uint32_t meta_i = 0;
    for (const ColumnMeta CM : *metadatas) {
        if (CM.type == CASS_VALUE_TYPE_INT) {
            parsers[meta_i] = new Int32Parser(CM);
        }
        else if (CM.type == CASS_VALUE_TYPE_BIGINT || CM.type == CASS_VALUE_TYPE_VARINT) {
            parsers[meta_i] = new Int64Parser(CM);
        }
        else if (CM.type == CASS_VALUE_TYPE_BOOLEAN) {
            parsers[meta_i] = new BoolParser(CM);
        }
        else if (CM.type == CASS_VALUE_TYPE_TEXT || CM.type == CASS_VALUE_TYPE_VARCHAR || CM.type ==  CASS_VALUE_TYPE_ASCII) {
            parsers[meta_i] = new TextParser(CM);
        }
        else if (CM.type == CASS_VALUE_TYPE_BLOB) {
            parsers[meta_i] = new BytesParser(CM);
        }
        else if (CM.type == CASS_VALUE_TYPE_DOUBLE || CM.type == CASS_VALUE_TYPE_FLOAT) {
            parsers[meta_i] = new DoubleParser(CM);
        }
        else if (CM.type == CASS_VALUE_TYPE_UUID) {
            parsers[meta_i] = new UuidParser(CM);
        }
        else if (CM.type == CASS_VALUE_TYPE_SMALL_INT) {
            parsers[meta_i] = new Int16Parser(CM);
        }
        else if (CM.type == CASS_VALUE_TYPE_TINY_INT) {
            parsers[meta_i] = new Int8Parser(CM);
        }
        else if (CM.type == CASS_VALUE_TYPE_UDT && CM.info.find("table") != CM.info.end()) {
            NumpyParser *NP = new NumpyParser(CM);
            NP->setStorage(storage);
            parsers[meta_i] = NP;
        }
        else parsers[meta_i]= new UnitParser(CM);
        ++meta_i;
    }
}

PythonParser::~PythonParser() {
    for (UnitParser *parser : this->parsers) {
        delete(parser);
    }
}
/*** TUPLE BUILDERS ***/

/***
 * Build a tuple from the given Python object using the stored metadata
 * @param obj Python List containing exactly the same number of objects that parsers/metadata sued
 * to setup the PythonParser
 * @return TupleRow with a copy of the values in obj
 * @post The python object can be garbage collected
 */
TupleRow *PythonParser::make_tuple(PyObject* obj) const {
    if (!PyList_Check(obj)) throw ModuleException("PythonParser: Make tuple: Expected python list");
    if (size_t(PyList_Size(obj))!=parsers.size())
        throw ModuleException("PythonParser: Got less python elements than columns configured");

    uint32_t total_bytes = metas->at(metas->size()-1).position+metas->at(metas->size()-1).size;
    char *buffer = (char *) malloc(total_bytes);
    TupleRow *new_tuple = new TupleRow(metas, total_bytes, buffer);

    for (uint32_t i = 0; i < PyList_Size(obj); ++i) {
        PyObject* some = PyList_GetItem(obj, i);
        if (this->parsers[i]->py_to_c(some, buffer+metas->at(i).position)<0) new_tuple->setNull(i);
    }
    return new_tuple;
}

/***
 * Builds a Python list from the data being held inside the TupleRow
 * @param tuple
 * @return A list with the information from tuple preserving its order
 */
PyObject *PythonParser::make_pylist(std::vector<const TupleRow *> &values) const {
    //TODO design behaviour, should we expect N tuples or just one?
    const TupleRow *tuple = values[0];
    if (tuple == nullptr)
        throw ModuleException("PythonParser: Marshalling from c to python a NULL tuple, unsupported");
    if (tuple->n_elem() != parsers.size())
        throw ModuleException("PythonParser: Found " + std::to_string(tuple->n_elem()) +
                              " elements from a max of " + std::to_string(parsers.size()));

    PyObject *list = PyList_New(tuple->n_elem());
    for (uint16_t i = 0; i < tuple->n_elem(); i++) {
        PyList_SetItem(list, i, this->parsers[i]->c_to_py(tuple->get_element(i)));
    }
    return list;
}
