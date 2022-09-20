#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

char * generateKey(int ts) {

    char * key = (char*) malloc (sizeof(int));
    *(int *)key = ts;
    return key;
}

#define ROWS 4
#define COLS 3
char * generateMetas() {
    unsigned int * metas = (unsigned int *) malloc(sizeof(unsigned int) * 3);

    metas[0]=2; // number of dimmensions
    metas[1]=ROWS; // number of elements in the first dimmension
    metas[2]=COLS; // number of elements in the second dimmension

    return (char *) metas;
}

char * generateNumpyContent() {

    double *numpy=(double*)malloc(sizeof(double)*COLS*ROWS);
    double *tmp = numpy;
    double num = 1;
    for (int i=0; i<ROWS; i++) {
        for (int j=0; j<COLS; j++) {
            std::cout<< "++ "<<i<<","<<j<<std::endl;
            *tmp = num++;
            tmp+=1;
        }
    }
    return (char*) numpy;
}

void dict_with_numpys(HecubaSession &s) {
    char * key;
    char * value;
    char * numpymeta;

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    IStorage* midict = s.createObject("miclassNumpy", "streaming_dict_with_numpy");
    std::cout<< "+ 'dict' object created"<<std::endl;

    // create a StorageNumpy and then add it to the StorageDict
    numpymeta = generateMetas();
    std::cout<< "+ metadata created"<<std::endl;
    value = generateNumpyContent();
    std::cout<< "+ value created at "<<std::hex<<(void*)value<<std::endl;

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    IStorage *mi_sn=s.createObject("hecuba.hnumpy.StorageNumpy","minp",numpymeta,value);
    mi_sn->sync(); // currently we do not support nested send, therefore the following setItem sends the uuid only and the consumer builds the object. To guarantee that the consumer can access the numpy, the producer needs to execute a sync after setting the element

    key = generateKey(42);
    std::cout<< "+  key created"<< std::endl;

    midict->setItem((void*)key, mi_sn); // currently we do not support nested send, this setItem sends the uuid only and the consumer builds the object. To guarantee that the consumer can access the numpy, the producer nees to execute a sync after setting the element. This only affects to 'IStorage *', the basic types are sent correctly.
    std::cout<< "+ value created at "<<std::hex<<(void*)mi_sn->getStorageID()<<std::endl;

    free(key);
    free(numpymeta);
    free(value);
    delete(midict); // this calls the destructor of the object that flushes any pending messages
    delete(mi_sn);
    std::cout<< "+ AFTER sync "<<std::endl;
}
void dict_with_string(HecubaSession &s) {

    // createObject executes a 'new', therefore reference MUST be deleted by the user
    IStorage* midict = s.createObject("miclass", "streaming_dict_with_str");
    std::cout<< "+ 'dict' object created"<<std::endl;

    int keyInt = 666;
    const char* value ="Oh! Yeah! Holidays!";
    midict->setItem((void*)&keyInt, &value);

    delete(midict); // this calls the destructor of the object that flushes any pending messages

    std::cout<< "+ AFTER sync "<<std::endl;
}

void subclass_storageNumpy(HecubaSession &s) {
    // create a StorageNumpy and then add it to the StorageDict
    void *numpymeta = generateMetas();
    std::cout<< "+ metadata created"<<std::endl;
    void *value = generateNumpyContent();

    std::cout<< "+ value created at "<<std::hex<<(void*)value<<std::endl;

    IStorage* minumpy = s.createObject("myNumpy", "i_am_a_numpy", numpymeta, value);
    std::cout<< "+ 'i_am_a_numpy' subclass of StorageNumpy object created" << std::endl;

    minumpy->send();
    delete(minumpy); // this calls the destructor of the object that flushes any pending messages
    std::cout<< "+ AFTER sync "<<std::endl;

}

int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;
    HecubaSession s;
    std::cout<< "+ Session started"<<std::endl;

    s.loadDataModel("hecuba_stream.yaml");
    std::cout<< "+ Data Model loaded"<<std::endl;

    dict_with_numpys(s);
    dict_with_string(s);
    subclass_storageNumpy(s);
    std::cout<< "++++ REMEMBER TO LAUNCH: python3 ./consumer.py to test the streaming results"<<std::endl;
}
