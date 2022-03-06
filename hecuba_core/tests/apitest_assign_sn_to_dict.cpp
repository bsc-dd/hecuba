#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

char * generateKey(float lat, int ts) {

    char * key = (char*) malloc (sizeof(float) + sizeof(int));
    float *lat_key = (float*) key;
    *lat_key = lat;
    int *ts_key = (int*) (key + sizeof(float));
    *ts_key = ts;
    std::cout << " generatekey sizeof(float) "<< sizeof(float) << " sizeof(int) " << sizeof(int)<< std::endl;
    return key;
}

#define ROWS 4
#define COLS 3
char * generateMetas() {
    unsigned int * metas = (unsigned int *) malloc(sizeof(unsigned int) * 3);

    metas[0]=2; // number of dimmensions
    metas[1]=ROWS; // number of elements in the first dimmension
    metas[2]=COLS; // number of elements in the second dimmension
    //metas[3]=sizeof(double); //'f' ONLY FLOATS SUPPORTED

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

int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;
    HecubaSession s;
    std::cout<< "+ Session started"<<std::endl;
    char * key;
    char * value;
    char * numpymeta;

    s.loadDataModel("model_class.yaml","model_class.py");
    std::cout<< "+ Data Model loaded"<<std::endl;

    std::cout<< "DEBUG " << s.getDataModel()->debug() << std::endl;

    IStorage* midict = s.createObject("midict", "yoli");
    std::cout<< "+ 'dict' object created"<<std::endl;

    // create a StorageNumpy and then add it to the StorageDict
    numpymeta = generateMetas();
    std::cout<< "+ metadata created"<<std::endl;
    value = generateNumpyContent();
    std::cout<< "+ value created at "<<std::hex<<(void*)value<<std::endl;


    IStorage *mi_sn=s.createObject("hecuba.hnumpy.StorageNumpy","minp",numpymeta,value);

    key = generateKey(0.5, 0);
    std::cout<< "+  key created"<< std::endl;

    // midict[key] = value;

    midict->setItem((void*)key, (void*) &mi_sn);
    std::cout<< "+ value created at "<<std::hex<<(void*)mi_sn->getStorageID()<<std::endl;
    midict->sync();

    std::cout<< "+ AFTER sync "<<std::endl;

    // Manually write the StorageNumpy using the write_to_cassandra method (we skip setitem)
    Writer *w = midict->getDataWriter();
    std::cout<< "+ AFTER getDataWriter "<<std::endl;

    key = generateKey(1.5, 0);

    uint32_t value_size = 2*sizeof(uint64_t);
    uint64_t* c_value_copy = (uint64_t*)malloc(value_size);
    std::memcpy(c_value_copy, mi_sn->getStorageID(), value_size);

    void * cc_val = malloc(sizeof(uint64_t*)); //uuid(numpy)
    std::memcpy((char *)cc_val, &c_value_copy, sizeof(uint64_t*));


    w->write_to_cassandra((void*)key, (void*) cc_val, "metrics");
    std::cout<< "+ AFTER write_to_cassandra "<<std::endl;
    midict->sync();

    std::cout<< "+ AFTER sync "<<std::endl;

    // Add a new item we create a new StorageNumpy in the setitem method
    key = generateKey(2.5, 1);
    midict->setItem((void*)key, &mi_sn);
    std::cout<< "+ AFTER setitem "<<std::endl;
    midict->sync();
    std::cout<< "+ AFTER sync "<<std::endl;

    free(key);
    free(numpymeta);
    free(value);

}
