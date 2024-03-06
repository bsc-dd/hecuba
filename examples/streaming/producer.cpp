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
	my_sn.make_persistent("mynp");

	IntKeyClass key(42);
    std::cout<< "+  key created"<< std::endl;

	NumpyValueClass my_value = NumpyValueClass(my_sn);
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

class myNumpy: public StorageNumpy, public StorageStream {
#ifdef OPCION1
public:
	myNumpy (void *data,const std::vector<uint32_t> &metas): StorageNumpy(data,metas){
	}
#endif
};

void subclass_storageNumpy() {
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
	my_sn.make_persistent("mynpsubclass");

	my_sn.sync();

    std::cout<< "+ 'StorageNumpy' object created"<<std::endl;

    my_sn.send();
    //delete(minumpy); // this calls the destructor of the object that flushes any pending messages
	
    std::cout<< "+ AFTER sending "<<std::endl;

}

int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;

    dict_with_numpys();
    dict_with_string();
    //subclass_storageNumpy(); //TODO: Generate automatically myNumpy.py
    std::cout<< "++++ REMEMBER TO LAUNCH: python3 ./consumer.py to test the streaming results"<<std::endl;
}
