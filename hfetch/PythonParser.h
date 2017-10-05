#ifndef PYTHON_PARSER_H
#define PYTHON_PARSER_H

#include <python2.7/Python.h>

#include <string>
#include <iostream>
#include <vector>
#include <stdlib.h>


#include "TupleRow.h"
#include "ModuleException.h"
#include "TableMetadata.h"
#include "StorageInterface.h"
#include "UnitParser.h"


class PythonParser {

public:
    PythonParser(std::shared_ptr<StorageInterface> storage, std::shared_ptr<const std::vector<ColumnMeta> > metadatas);

    ~PythonParser();

    TupleRow *make_tuple(PyObject *obj) const;

    PyObject *make_pylist(std::vector<const TupleRow *> &values) const;

private:
    std::vector<UnitParser *> parsers;
    std::shared_ptr<const std::vector<ColumnMeta> > metas; //TODO To be removed
};


#endif //PYTHON_PARSER_H
