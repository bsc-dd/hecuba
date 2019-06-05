#include <iostream>
#include "gtest/gtest.h"


#include "../src/py_interface/PythonParser.h"

using namespace std;

#define PY_ERR_CHECK if (PyErr_Occurred()){PyErr_Print(); PyErr_Clear();}


/** TEST SETUP **/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Py_Initialize();

    PyEval_InitThreads();

    return RUN_ALL_TESTS();

}



/** PYTHON INTERFACE TESTS **/


#define Py_STRING "s"
#define Py_U_LONGLONG "K"
#define Py_U_LONG "k"
#define Py_LONGLONG "L"
#define Py_LONG "l"
#define Py_BOOL "b"
#define Py_INT "i"
#define Py_U_INT "I"
#define Py_FLOAT "f"
#define Py_DOUBLE "d"
#define Py_SHORT_INT "h"

/** Test to verify Python doubles parsing is performing as expected **/
TEST(TestPyParse, DoubleParse) {
    PyErr_Clear();

    PyObject *key = PyLong_FromDouble(0.12);
    int ok = 0;

    size_t data_size = sizeof(cass_double_t);
    void *data_a = malloc(data_size);
    void *data_b = malloc(data_size);

    cass_double_t t;
    ok = PyArg_Parse(key, Py_DOUBLE, &t);
    EXPECT_EQ(ok, 1);
    memcpy(data_a, &t, sizeof(t));
    ok = PyArg_Parse(key, Py_DOUBLE, data_b);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(memcmp(data_a, data_b, data_size), 0);
    std::free(data_a);
    free(data_b);
    PY_ERR_CHECK
}



/** Numpy parse and library test **/

TEST(TestPythonBlob, TupleRowParsing) {
    /** setup test **/
    PyErr_Clear();
    /*
    if (_import_array() < 0) {
         PyErr_Print();
         PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
         NUMPY_IMPORT_ARRAY_RETVAL;
     }
     npy_intp dims[2] = {2, 2};
     void *array = malloc(sizeof(double) * 4);

     double *temp = (double *) array;
     *temp = 123;
     *(temp + 1) = 456;
     *(temp + 2) = 789;
     *(temp + 3) = 500;
     PyObject *key = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, array);

 // interface receives key

     PyArrayObject *arr;
     int ok = PyArray_OutputConverter(key, &arr);
     if (!ok) throw ModuleException("error parsing PyArray to obj");

 // transform to bytes
     PyObject *bytes = PyArray_ToString(arr, NPY_KEEPORDER);
     PY_ERR_CHECK

     ok = PyString_Check(bytes);
     PY_ERR_CHECK
     Py_ssize_t l_size = PyString_Size(bytes);
     PY_ERR_CHECK

 // store bytes
     void *data = malloc(l_size);
     char *l_temp = PyString_AsString(bytes);
     PY_ERR_CHECK
     char *permanent = (char *) malloc(l_size + 1);
     memcpy(permanent, l_temp, l_size);
     permanent[l_size] = '\0';
     memcpy(data, &permanent,sizeof(char *));
     PY_ERR_CHECK

 // cleanup
     Py_DecRef(key);

     free(data);
     free(permanent);*/
}


/*** UNIT PARSERS ***/

TEST(TestPythonUnitParsers, ParseInt16) {
    PyErr_Clear();
    int32_t value;
    int16_t result, ok, range = 20000; //32768 is the max value 2^15
    PyObject *py_int;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_SMALL_INT;
    uint16_t offset = 0;
    uint16_t bsize = sizeof(int16_t);
    ColumnMeta CM = ColumnMeta(info, cv_type, offset, bsize);
    UnitParser *parser = new Int16Parser(CM);

    for (value = -(range / 2); value < (range / 2); ++value) {
        py_int = Py_BuildValue(Py_SHORT_INT, value);
        EXPECT_FALSE(py_int == nullptr);
        ok = parser->py_to_c(py_int, &result);
        EXPECT_FALSE(ok == -1); //object was null
        EXPECT_FALSE(ok == -2); //something went wrong
        EXPECT_TRUE(ok == 0); //it worked as expected
        EXPECT_EQ(result, value);
    }
    delete (parser);
}


/*TEST(TestPythonUnitParsers, ParseInt32) {
    PyErr_Clear();
    int32_t result, value, ok, range = 20000;
    PyObject *py_int;
    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_INT;
    uint16_t offset = 0;
    uint16_t bsize = sizeof(int32_t);
    ColumnMeta CM = ColumnMeta(info, cv_type, offset, bsize);
    UnitParser *parser = new Int32Parser(CM);

    for (value = -(range / 2); value < (range / 2); ++value) {
        py_int = PyInt_FromLong(value);
        ok = parser->py_to_c(py_int, &result);
        EXPECT_FALSE(ok == -1); //object was null
        EXPECT_FALSE(ok == -2); //something went wrong
        EXPECT_TRUE(ok == 0); //it worked as expected
        EXPECT_EQ(result, value);
    }
    delete (parser);
}*/


TEST(TestPythonUnitParsers, ParseInt64BigInt) {
    PyErr_Clear();
    int64_t result, value, range = 20000;
    int16_t ok;
    PyObject *py_int;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_BIGINT;
    uint16_t offset = 0;
    uint16_t bsize = sizeof(int64_t);
    ColumnMeta CM = ColumnMeta(info, cv_type, offset, bsize);
    UnitParser *parser = new Int64Parser(CM);

    for (value = -(range / 2); value < (range / 2); ++value) {
        py_int = PyLong_FromLong(value);
        ok = parser->py_to_c(py_int, &result);
        EXPECT_FALSE(ok == -1); //object was null
        EXPECT_FALSE(ok == -2); //something went wrong
        EXPECT_TRUE(ok == 0); //it worked as expected
        EXPECT_EQ(result, value);
    }
    delete (parser);
}


TEST(TestPythonUnitParsers, ParseInt64VarInt) {
    PyErr_Clear();
    int64_t result, value, range = 20000;
    int16_t ok;
    PyObject *py_int;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_VARINT;
    uint16_t offset = 0;
    uint16_t bsize = sizeof(int64_t);
    ColumnMeta CM = ColumnMeta(info, cv_type, offset, bsize);
    UnitParser *parser = new Int64Parser(CM);

    for (value = -(range / 2); value < (range / 2); ++value) {
        py_int = PyLong_FromLong(value);
        ok = parser->py_to_c(py_int, &result);
        EXPECT_FALSE(ok == -1); //object was null
        EXPECT_FALSE(ok == -2); //something went wrong
        EXPECT_TRUE(ok == 0); //it worked as expected
        EXPECT_EQ(result, value);
    }
    delete (parser);
}


TEST(TestPythonUnitParsers, ParseDouble) {
    PyErr_Clear();
    double result, value, range = 100.0;
    int16_t ok;
    PyObject *py_int;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_DOUBLE;
    uint16_t offset = 0;
    uint16_t bsize = sizeof(double);
    ColumnMeta CM = ColumnMeta(info, cv_type, offset, bsize);
    UnitParser *parser = new DoubleParser(CM);

    for (value = -(range / 2); value < (range / 2); value += 0.1) {
        py_int = PyFloat_FromDouble(value);
        ok = parser->py_to_c(py_int, &result);
        EXPECT_FALSE(ok == -1); //object was null
        EXPECT_FALSE(ok == -2); //something went wrong
        EXPECT_TRUE(ok == 0); //it worked as expected
        EXPECT_EQ(result, value);
    }
    delete (parser);
}


TEST(TestPythonUnitParsers, ParseFloat) {
    PyErr_Clear();
    float result, range = 100.0;
    double value;
    int16_t ok;
    PyObject *py_int;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_FLOAT;
    uint16_t offset = 0;
    uint16_t bsize = sizeof(float);
    ColumnMeta CM = ColumnMeta(info, cv_type, offset, bsize);
    UnitParser *parser = new DoubleParser(CM);

    for (value = -(range / 2); value < (range / 2); value += 0.1) {
        py_int = PyFloat_FromDouble(value);
        ok = parser->py_to_c(py_int, &result);
        EXPECT_FALSE(ok == -1); //object was null
        EXPECT_FALSE(ok == -2); //something went wrong
        EXPECT_TRUE(ok == 0); //it worked as expected
        EXPECT_FLOAT_EQ(result, value);
    }
    delete (parser);
}

TEST(TestPythonUnitParsers, ParseTuple_py_to_c_INT) {
    PyErr_Clear();


    int32_t value, ok = 20000;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(int));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_INT, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_INT, bsize, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    UnitParser *parser = new TupleParser(CM);

    PyObject *pt = Py_BuildValue("(ii)", 4, 5);

    void *result = malloc(sizeof(int32_t) * 2);
    void *external = malloc(sizeof(TupleRow *));
    ok = parser->py_to_c(pt, external);

    const TupleRow *inner_data = *reinterpret_cast<const TupleRow **>(external);
    const void *elem = inner_data->get_element(0);
    const int32_t uziv1 = *(int32_t const *) elem;
    const void *elem1 = inner_data->get_element(1);
    const int32_t uziv2 = *(int32_t const *) elem1;

    EXPECT_EQ(uziv1, 4);
    EXPECT_EQ(uziv2, 5);

    EXPECT_FALSE(ok == -1); //object was null
    EXPECT_FALSE(ok == -2); //something went wrong
    EXPECT_TRUE(ok == 0); //it worked as expected

    PyObject *tuple = parser->c_to_py(external);
    PyObject *result1 = PyTuple_GetItem(tuple, 0);
    long p1 = PyLong_AsLong(result1);
    EXPECT_EQ(p1, 4);
    result1 = PyTuple_GetItem(tuple, 1);
    p1 = PyLong_AsLong(result1);
    EXPECT_EQ(p1, 5);

}

TEST(TestPythonUnitParsers, ParseTuple_py_to_c_LONG) {
    PyErr_Clear();


    int32_t value, ok = 20000;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(int64_t));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_BIGINT, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_BIGINT, bsize, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    UnitParser *parser = new TupleParser(CM);

    PyObject *pt = Py_BuildValue("(LL)", 5500000000000000L, 9223372036854775806);
    void *external = malloc(sizeof(TupleRow *));
    ok = parser->py_to_c(pt, external);

    const TupleRow *inner_data = *reinterpret_cast<const TupleRow **>(external);
    const void *elem = inner_data->get_element(0);
    const int64_t uziv1 = *(int64_t const *) elem;
    const void *elem1 = inner_data->get_element(1);
    const int64_t uziv2 = *(int64_t const *) elem1;


    EXPECT_FALSE(ok == -1); //object was null
    EXPECT_FALSE(ok == -2); //something went wrong
    EXPECT_TRUE(ok == 0); //it worked as expected

    PyObject *tuple = parser->c_to_py(external);
    PyObject *result1 = PyTuple_GetItem(tuple, 0);
    long p1 = PyLong_AsLong(result1);
    EXPECT_EQ(p1, 5500000000000000L);
    result1 = PyTuple_GetItem(tuple, 1);
    p1 = PyLong_AsLong(result1);
    EXPECT_EQ(p1, 9223372036854775806);

}

TEST(TestPythonUnitParsers, ParseTuple_py_to_c_TEXT) {
    PyErr_Clear();


    int32_t value, ok = 20000;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(int64_t));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_TEXT, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_TEXT, bsize, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    UnitParser *parser = new TupleParser(CM);

    PyObject *pt = Py_BuildValue("(ss)", "text1", "text2");
    void *external = malloc(sizeof(TupleRow *));
    ok = parser->py_to_c(pt, external);

    const TupleRow *inner_data = *reinterpret_cast<const TupleRow **>(external);
    const void *elem = inner_data->get_element(0);
    const int64_t uziv1 = *(int64_t const *) elem;
    const void *elem1 = inner_data->get_element(1);
    const int64_t uziv2 = *(int64_t const *) elem1;


    EXPECT_FALSE(ok == -1); //object was null
    EXPECT_FALSE(ok == -2); //something went wrong
    EXPECT_TRUE(ok == 0); //it worked as expected

    PyObject *tuple = parser->c_to_py(external);
    PyObject *result1 = PyTuple_GetItem(tuple, 0);
    std::string valstr = PyString_AsString(result1);
    EXPECT_EQ(valstr, "text1");
    result1 = PyTuple_GetItem(tuple, 1);
    valstr = PyString_AsString(result1);
    EXPECT_EQ(valstr, "text2");


}

TEST(TestPythonUnitParsers, ParseTuple_py_to_c_DOUBLE) {
    PyErr_Clear();


    int32_t value, ok = 20000;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(float));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_FLOAT, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_FLOAT, bsize, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    UnitParser *parser = new TupleParser(CM);

    PyObject *pt = Py_BuildValue("(dd)", 2.00, 2.01);
    void *external = malloc(sizeof(TupleRow *));
    ok = parser->py_to_c(pt, external);

    const TupleRow *inner_data = *reinterpret_cast<const TupleRow **>(external);
    const void *elem = inner_data->get_element(0);
    const int64_t uziv1 = *(int64_t const *) elem;
    const void *elem1 = inner_data->get_element(1);
    const int64_t uziv2 = *(int64_t const *) elem1;


    EXPECT_FALSE(ok == -1); //object was null
    EXPECT_FALSE(ok == -2); //something went wrong
    EXPECT_TRUE(ok == 0); //it worked as expected

    PyObject *tuple = parser->c_to_py(external);
    PyObject *result1 = PyTuple_GetItem(tuple, 0);
    float res = PyFloat_AsDouble(result1);
    EXPECT_FLOAT_EQ(res, 2.00);
    result1 = PyTuple_GetItem(tuple, 1);
    res = PyFloat_AsDouble(result1);
    EXPECT_FLOAT_EQ(res, 2.01);

}

TEST(TestPythonUnitParsers, ParseTuple_py_to_c_BOOLEAN) {
    PyErr_Clear();


    int32_t value, ok = 20000;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(bool));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_BOOLEAN, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_BOOLEAN, bsize, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    UnitParser *parser = new TupleParser(CM);
    PyObject *t1 = Py_True;
    PyObject *t2 = Py_False;
    PyObject *pt = Py_BuildValue("(OO)", t1, t2);
    void *external = malloc(sizeof(TupleRow *));
    ok = parser->py_to_c(pt, external);

    const TupleRow *inner_data = *reinterpret_cast<const TupleRow **>(external);
    const void *elem = inner_data->get_element(0);
    const int64_t uziv1 = *(int64_t const *) elem;
    const void *elem1 = inner_data->get_element(1);
    const int64_t uziv2 = *(int64_t const *) elem1;


    EXPECT_FALSE(ok == -1); //object was null
    EXPECT_FALSE(ok == -2); //something went wrong
    EXPECT_TRUE(ok == 0); //it worked as expected

    PyObject *tuple = parser->c_to_py(external);
    PyObject *result1 = PyTuple_GetItem(tuple, 0);
    EXPECT_EQ(result1, t1);
    result1 = PyTuple_GetItem(tuple, 1);
    EXPECT_EQ(result1, t2);

}

TEST(TestPythonUnitParsers, ParseTuple_py_to_c_DOUBLE_AND_TEXT) {
    PyErr_Clear();


    int32_t value, ok = 20000;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(float));
    uint16_t bsize2 = (sizeof(int64_t));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_FLOAT, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_TEXT, bsize, bsize2);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    UnitParser *parser = new TupleParser(CM);

    PyObject *pt = Py_BuildValue("(ds)", 2.00, "hola");
    void *external = malloc(sizeof(TupleRow *));
    ok = parser->py_to_c(pt, external);

    const TupleRow *inner_data = *reinterpret_cast<const TupleRow **>(external);
    const void *elem = inner_data->get_element(0);
    const int64_t uziv1 = *(int64_t const *) elem;
    const void *elem1 = inner_data->get_element(1);
    const int64_t uziv2 = *(int64_t const *) elem1;


    EXPECT_FALSE(ok == -1); //object was null
    EXPECT_FALSE(ok == -2); //something went wrong
    EXPECT_TRUE(ok == 0); //it worked as expected

    PyObject *tuple = parser->c_to_py(external);
    PyObject *result1 = PyTuple_GetItem(tuple, 0);
    float res = PyFloat_AsDouble(result1);
    EXPECT_FLOAT_EQ(res, 2.00);
    result1 = PyTuple_GetItem(tuple, 1);
    std::string valstr = PyString_AsString(result1);
    EXPECT_EQ(valstr, "hola");

}

TEST(TestPythonUnitParsers, ParseTuple_c_to_py_CHECK_NULL) {
    PyErr_Clear();


    int32_t value, ok = 20000;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(float));
    uint16_t bsize2 = (sizeof(int64_t));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_FLOAT, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_TEXT, bsize, bsize2);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    UnitParser *parser = new TupleParser(CM);

    PyObject *pt = Py_BuildValue("(ds)", 2.1, "hola");
    void *external = malloc(sizeof(TupleRow *));
    ok = parser->py_to_c(pt, external);

    TupleRow *inner_data = *reinterpret_cast< TupleRow **>(external);
    const void *elem = inner_data->get_element(0);
    const int64_t uziv1 = *(int64_t const *) elem;
    const void *elem1 = inner_data->get_element(1);
    const int64_t uziv2 = *(int64_t const *) elem1;

    inner_data->setNull(0);

    EXPECT_FALSE(ok == -1); //object was null
    EXPECT_FALSE(ok == -2); //something went wrong
    EXPECT_TRUE(ok == 0); //it worked as expected

    PyObject *tuple = parser->c_to_py(external);
    PyObject *result1 = PyTuple_GetItem(tuple, 0);
    EXPECT_EQ(result1, Py_None);
    result1 = PyTuple_GetItem(tuple, 1);
    std::string valstr = PyString_AsString(result1);
    EXPECT_EQ(valstr, "hola");

}

TEST(TestPythonUnitParsers, ParseTuple_py_to_c_CHECK_NULL) {
    PyErr_Clear();


    int32_t value, ok = 20000;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(int32_t));
    uint16_t bsize2 = (sizeof(int64_t));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_INT, 0, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_TEXT, bsize, bsize2);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(TupleRow *);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    UnitParser *parser = new TupleParser(CM);

    PyObject *pt = Py_BuildValue("(Os)", Py_None, "hola");
    void *external = malloc(sizeof(TupleRow *));
    ok = parser->py_to_c(pt, external);

    TupleRow *inner_data = *reinterpret_cast< TupleRow **>(external);
    const void *elem = inner_data->get_element(0);
    const void *elem1 = inner_data->get_element(1);
    const int64_t uziv2 = *(int64_t const *) elem1;

    EXPECT_FALSE(ok == -1); //object was null
    EXPECT_FALSE(ok == -2); //something went wrong
    EXPECT_TRUE(ok == 0); //it worked as expected

    PyObject *tuple = parser->c_to_py(external);
    PyObject *result1 = PyTuple_GetItem(tuple, 0);
    EXPECT_EQ(result1, Py_None);
    result1 = PyTuple_GetItem(tuple, 1);
    std::string valstr = PyString_AsString(result1);
    EXPECT_EQ(valstr, "hola");

}
