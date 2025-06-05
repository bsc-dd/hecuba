#include <StorageDict.h>
#include <StorageNumpy.h>
#include <StorageStream.h>
#include <KeyClass.h>
#include <ValueClass.h>
#include <iostream>

char * generateKey(int ts) {

    char * key = (char*) malloc (sizeof(int));
    *(int *)key = ts;
    return key;
}

char * generateNumpyContent(const std::vector<uint32_t> &metas) {

    double *numpy=(double*)malloc(sizeof(double)*metas[0]*metas[1]);
    double *tmp = numpy;
    double num = 1;
    for (int i=0; i<metas[0]; i++) {
        for (int j=0; j<metas[1]; j++) {
            std::cout<< "++ "<<i<<","<<j<<std::endl;
            *tmp = num++;
            tmp+=1;
        }
    }
    std::cout<< "+ Generated NUMPY ["<<metas[0]<<", "<<metas[1]<<"] using "<<sizeof(double)*metas[0]*metas[1]<<"bytes at "<<std::hex<<(void*)numpy<<std::endl;
    return (char*) numpy;
}

using IntKeyClass = KeyClass<int32_t>;
using NumpyValueClass = ValueClass<StorageNumpy>;

class DictWithNumpy: public StorageDict <IntKeyClass, NumpyValueClass, DictWithNumpy>, public StorageStream {

};

#define ROWS 3
#define COLS 4

void dict_with_numpys() {
	
	std::string dictName = "streaming_dict_with_numpy";
	
	DictWithNumpy mydict;

	mydict.make_persistent( dictName );
    std::cout<< "+ Dictionary "<<dictName<< " object created"<<std::endl;

    // create a StorageNumpy and then add it to the StorageDict
    std::vector<uint32_t> metadata = {ROWS, COLS};
    char* data = generateNumpyContent(metadata);
    std::cout<< "+ value created at "<<std::hex<<(void*)data<<std::endl;

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    StorageNumpy my_sn(data, metadata);
	my_sn.make_persistent("mynpDict");

	IntKeyClass key(42);
    std::cout<< "+  key created"<< std::endl;

	NumpyValueClass my_value = NumpyValueClass(my_sn);
    mydict[key] = my_value;

    std::cout<< "+ value created at "<<std::hex<<(void*)my_sn.getStorageID()<<std::endl;

    std::cout<< "+ AFTER sync "<<std::endl;
}

using MultAttrValueClass=ValueClass<int32_t,StorageNumpy>;
class DictWithMultValue: public StorageDict <IntKeyClass,MultAttrValueClass,DictWithMultValue>, public StorageStream {};
void dict_with_multiple_values() {

	std::string dictName = "streaming_dict_with_multiplevalues";

	DictWithMultValue mydict;

	mydict.make_persistent( dictName );
    std::cout<< "+ Dictionary "<<dictName<< " object created"<<std::endl;

    // create a StorageNumpy and then add it to the StorageDict
    std::vector<uint32_t> metadata = {ROWS, COLS};
    char* data = generateNumpyContent(metadata);
    std::cout<< "+ value created at "<<std::hex<<(void*)data<<std::endl;

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    StorageNumpy my_sn(data, metadata);
	my_sn.make_persistent("mynpMulVal");

	IntKeyClass key(42);
    std::cout<< "+  key created"<< std::endl;

	MultAttrValueClass my_value = MultAttrValueClass(43,my_sn);
    mydict[key] = my_value;

    std::cout<< "+ value created at "<<std::hex<<(void*)my_sn.getStorageID()<<std::endl;

    std::cout<< "+ AFTER sync "<<std::endl;
}

using MultAttrValueClass2=ValueClass<StorageNumpy,int32_t>;
class DictWithMultValue2: public StorageDict <IntKeyClass,MultAttrValueClass2,DictWithMultValue2> , public StorageStream {};
void dict_with_multiple_values2() {

	std::string dictName = "streaming_dict_with_multiplevalues2";

	DictWithMultValue2 mydict;

	mydict.make_persistent( dictName );
    std::cout<< "+ Dictionary "<<dictName<< " object created"<<std::endl;

    // create a StorageNumpy and then add it to the StorageDict
    std::vector<uint32_t> metadata = {ROWS, COLS};
    char* data = generateNumpyContent(metadata);
    std::cout<< "+ value created at "<<std::hex<<(void*)data<<std::endl;

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    StorageNumpy my_sn(data, metadata);
	my_sn.make_persistent("mynpMulVal2");

	IntKeyClass key(42);
    std::cout<< "+  key created"<< std::endl;

	MultAttrValueClass2 my_value = MultAttrValueClass2(my_sn,43);
    mydict[key] = my_value;

    std::cout<< "+ value created at "<<std::hex<<(void*)my_sn.getStorageID()<<std::endl;

    std::cout<< "+ AFTER sync "<<std::endl;
}

using IntKeyClass = KeyClass<int32_t>;
using StringValueClass = ValueClass<std::string>;

//class DictWithStrings: public StorageDict <IntKeyClass, StringValueClass>, public StorageStream {
class DictWithStrings: public StorageDict <IntKeyClass, StringValueClass, DictWithStrings>,public StorageStream{

};
void dict_with_string() {

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    DictWithStrings midict;
	midict.make_persistent("streaming_dict_with_str");
    std::cout<< "+ 'dict' object created"<<std::endl;

    IntKeyClass keyInt ( 666 );
    StringValueClass value ("Oh! Yeah! Holidays!");
    midict[keyInt] = value;
    std::cout<< "+ AFTER setitem "<<std::endl;

    std::cout<< "+ AFTER sync "<<std::endl;
}

using IntKeyClass = KeyClass<int32_t>;
using MultipleBasicTypesValueClass = ValueClass<int32_t, std::string>;

class DictWithMultipleBasicTypes: public StorageDict <IntKeyClass, MultipleBasicTypesValueClass, DictWithMultipleBasicTypes>,public StorageStream{

};
void dict_with_multiple_basic_types() {

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    DictWithMultipleBasicTypes midict;
	midict.make_persistent("streaming_dict_with_multibasicvalues");
    std::cout<< "+ 'dict' object created"<<std::endl;

    IntKeyClass keyInt ( 666 );
    MultipleBasicTypesValueClass value (42, "Oh! Yeah! Holidays!");
    midict[keyInt] = value;
    std::cout<< "+ AFTER setitem "<<std::endl;

    std::cout<< "+ AFTER sync "<<std::endl;
}


class myNumpy: public StorageNumpy, public StorageStream {
#ifdef OPCION1
public:
	myNumpy (void *data,const std::vector<uint32_t> &metas): StorageNumpy(data,metas){
	}
#endif
};

void subclass_storageNumpy(const std::string& tablename) {
    // createObject executes a 'new', therefore reference MUST be deleted by the user

    std::vector<uint32_t> metadata = {ROWS, COLS};
    char* data = generateNumpyContent(metadata);
    std::cout<< "+ numpy content created at "<<std::hex<<(void*)data<<std::endl;

#ifdef OPCION1
    myNumpy my_sn (data, metadata);
#else 
    myNumpy my_sn;
	my_sn.setNumpy(data,metadata,'f');
#endif
	my_sn.make_persistent(tablename.c_str());

	my_sn.sync();

    std::cout<< "+ 'StorageNumpy' object created"<<std::endl;

    my_sn.send();
    //delete(minumpy); // this calls the destructor of the object that flushes any pending messages
	
    std::cout<< "+ AFTER sending "<<std::endl;

}

int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;

    dict_with_numpys();
    dict_with_multiple_basic_types();
    dict_with_multiple_values();
    dict_with_multiple_values2();
    dict_with_string();
    subclass_storageNumpy(std::string("mynpsubclass")); //TODO: Generate automatically myNumpy.py
    subclass_storageNumpy(std::string("mynpsubclass2")); //TODO: Generate automatically myNumpy.py
    std::cout<< "++++ REMEMBER TO LAUNCH: python3 ./consumer.py to test the streaming results"<<std::endl;
}
