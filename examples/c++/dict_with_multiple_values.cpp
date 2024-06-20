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
    return (char*) numpy;
}

using IntKeyClass = KeyClass<int32_t>;



#define ROWS 3
#define COLS 4

using MultAttrValueClass=ValueClass<int32_t,StorageNumpy>;
class DictWithMultValue: public StorageDict <IntKeyClass,MultAttrValueClass,DictWithMultValue> {};
void dict_with_multiple_values() {
	
	std::string dictName = "dict_with_multiplevalues";
	
	DictWithMultValue mydict;

	mydict.make_persistent( dictName );
    std::cout<< "+ Dictionary "<<dictName<< " object created"<<std::endl;

    // create a StorageNumpy and then add it to the StorageDict
    std::vector<uint32_t> metadata = {ROWS, COLS};
    char* data = generateNumpyContent(metadata);
    std::cout<< "+ value created at "<<std::hex<<(void*)data<<std::endl;

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    StorageNumpy my_sn(data, metadata);
	my_sn.make_persistent("mynp");

	IntKeyClass key(42);
    std::cout<< "+  key created"<< std::endl;

	MultAttrValueClass my_value = MultAttrValueClass(43,my_sn);
    mydict[key] = my_value;

    std::cout<< "+ value created at "<<std::hex<<(void*)my_sn.getStorageID()<<std::endl;

    std::cout<< "+ AFTER sync "<<std::endl;
}

using MultAttrValueClass2=ValueClass<StorageNumpy,int32_t>;
class DictWithMultValue2: public StorageDict <IntKeyClass,MultAttrValueClass2,DictWithMultValue2> {};
void dict_with_multiple_values2() {
	
	std::string dictName = "dict_with_multiplevalues2";
	
	DictWithMultValue2 mydict;

	mydict.make_persistent( dictName );
    std::cout<< "+ Dictionary "<<dictName<< " object created"<<std::endl;

    // create a StorageNumpy and then add it to the StorageDict
    std::vector<uint32_t> metadata = {ROWS, COLS};
    char* data = generateNumpyContent(metadata);
    std::cout<< "+ value created at "<<std::hex<<(void*)data<<std::endl;

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    StorageNumpy my_sn(data, metadata);
	my_sn.make_persistent("mynp");

	IntKeyClass key(42);
    std::cout<< "+  key created"<< std::endl;

	MultAttrValueClass2 my_value = MultAttrValueClass2(my_sn,43);
    mydict[key] = my_value;

    std::cout<< "+ value created at "<<std::hex<<(void*)my_sn.getStorageID()<<std::endl;

    std::cout<< "+ AFTER sync "<<std::endl;
}


int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;

    dict_with_multiple_values();
    dict_with_multiple_values2();
}
