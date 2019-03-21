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


TEST(TestPythonUnitParsers, ParseInt32) {
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
}


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

TEST(TestPythonUnitParsers, ParseTuple_py_to_c) {
    PyErr_Clear();

    int32_t result, value, ok = 20000;

    std::map<std::string, std::string> info = {{"name", "mycolumn"}};
    CassValueType cv_type = CASS_VALUE_TYPE_TUPLE;
    uint16_t offset = 0;
    uint16_t bsize = (sizeof(int));
    ColumnMeta cm1 = ColumnMeta(info, CASS_VALUE_TYPE_INT, offset, bsize);
    ColumnMeta cm2 = ColumnMeta(info, CASS_VALUE_TYPE_INT, offset, bsize);

    std::vector<ColumnMeta> v = {cm1, cm2};

    ColumnMeta CM = ColumnMeta();
    CM.info = {{"name", "ciao"}};
    CM.type = CASS_VALUE_TYPE_TUPLE;
    CM.position = 0;
    CM.size = sizeof(uint16_t);
    CM.pointer = std::make_shared<std::vector<ColumnMeta>>(v);

    UnitParser *parser = new TupleParser(CM);

    PyObject *pt = Py_BuildValue("(ii)", 4, 5);

    ok = parser->py_to_c(pt, &result);

    EXPECT_FALSE(ok == -1); //object was null
    EXPECT_FALSE(ok == -2); //something went wrong
    EXPECT_TRUE(ok == 0); //it worked as expected
    //EXPECT_EQ(result, value);


////////////////
    std::tuple<int,int> mytuple (10,20);

    char *buffer2 = (char *) malloc(sizeof(mytuple)); //values

    memcpy(buffer2, &mytuple, sizeof(mytuple));

    int *b = (int *)malloc(2*sizeof(int));

    char *p = buffer2;

    cout << "Els elements de la tupla son: " << endl;
    for(int i=0;i<2;i++) {
        b[i]=(int )*p;
        printf("got value %d\n",b[i]);
        p += sizeof(int);
    }

    TupleRow *values = new TupleRow(CM.pointer, sizeof(mytuple), buffer2);
    PyObject* tuple = PyTuple_New(2);
    tuple = parser->c_to_py(values);

    PyObject *result1 = PyTuple_GetItem(tuple, 0);
    cout << "El primer value del pyobj es: ";
    PyObject_Print(result1, stdout, 0);
    cout << endl;
    PyObject *result2 = PyTuple_GetItem(tuple, 1);
    cout << "El segon value del pyobj es: ";
    PyObject_Print(result2, stdout, 0);
    cout << endl;



    //EXPECT_EQ(result, value);

}

