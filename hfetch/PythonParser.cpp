#include "PythonParser.h"

/***
 * Builds a tuple factory to retrieve tuples based on rows and keys
 * extracting the information from Cassandra to decide the types to be used
 * @param table_meta Holds the table information
 */
PythonParser::PythonParser(std::shared_ptr<const std::vector<ColumnMeta> > metadatas) {
    this->parsers = std::vector<InnerParser*>(metadatas->size());
    for (uint32_t meta_i = 0; meta_i<metadatas->size(); ++meta_i) {
        if (metadatas->at(meta_i).type==CASS_VALUE_TYPE_INT) parsers[meta_i]=new Int32Parser(metadatas->at(meta_i));
        else if (metadatas->at(meta_i).type==CASS_VALUE_TYPE_TEXT) parsers[meta_i] = new TextParser(metadatas->at(meta_i));
        else parsers[meta_i]= new InnerParser(metadatas->at(meta_i));
    }
    this->metas = metadatas;
}

PythonParser::~PythonParser() {
    for (InnerParser *parser : this->parsers) {
        delete(parser);
    }

}
/*** TUPLE BUILDERS ***/

/***
 * Build a tuple from the given Python object using the factory's metadata
 * @param obj Python List containing exactly the same number of objects that metadata size
 * @return TupleRow with a copy of the values in obj
 * @post The python object can be deleted
 */
TupleRow *PythonParser::make_tuple(PyObject* obj) const {
    if (!PyList_Check(obj)) throw ModuleException("PythonParser: Make tuple: Expected python list");
    const std::vector<ColumnMeta>* localMeta=metas.get();
    if (size_t(PyList_Size(obj))!=localMeta->size())
        throw ModuleException("PythonParser: Got less python elements than columns configured");
    uint32_t total_bytes = localMeta->at(localMeta->size()-1).position+localMeta->at(localMeta->size()-1).size;

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

    PyObject *list;
    //store to cache
    const TupleRow *tuple = values[0];
    if (tuple == 0)
        throw ModuleException("TupleRowFactory: Marshalling from c to python a NULL tuple, unsupported");
    list = PyList_New(tuple->n_elem());
    for (uint16_t i = 0; i < tuple->n_elem(); i++) {
        if (i >= metas->size())
            throw ModuleException("TupleRowFactory: Tuple as py access meta at " + std::to_string(i) +
                                          " from a max " + std::to_string(metas->size()));
        PyList_SetItem(list, i, this->parsers[i]->c_to_py(tuple->get_element(i)));
    }
    return list;

}

int16_t PythonParser::InnerParser::py_to_c(PyObject* element,void* payload) const{
    throw ModuleException("Not implemented");
}

PyObject* PythonParser::InnerParser::c_to_py(const void* payload) const{
    throw ModuleException("Not implemented");
}
