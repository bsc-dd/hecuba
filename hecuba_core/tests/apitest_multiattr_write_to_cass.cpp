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

    s.loadDataModel("model_multiattr.yaml","model_multiattr.py");
    std::cout<< "+ Data Model loaded"<<std::endl;

    std::cout<< "DEBUG " << s.getDataModel()->debug() << std::endl;

    IStorage* midict = s.createObject("midict", "yoli");
    std::cout<< "+ 'dict' object created"<<std::endl;

    key = generateKey(0.5, 0);
    std::cout<< "+  key created"<< std::endl;

    Writer *w = midict->getDataWriter();
    std::cout<< "+ AFTER getDataWriter "<<std::endl;

    int v1_value = 0x1001;

    void * cc_val = malloc(sizeof(int32_t));

    std::memcpy((char *)cc_val, &v1_value, sizeof(int32_t));
    w->write_to_cassandra((void*)key, (void*) cc_val, "v1");

    std::cout<< "+ AFTER write_to_cassandra v1"<<std::endl;
    midict->sync();

    std::cout<< "+ AFTER sync "<<std::endl;
    ///////////////////////////////////////////// 
    key = generateKey(0.5, 0);
    std::cout<< "+  key created"<< std::endl;

    cc_val = malloc(sizeof(int32_t));
    int v2_value=0x42;
    std::memcpy((char *)cc_val, &v2_value, sizeof(int32_t));
    w->write_to_cassandra((void*)key, (void*) cc_val, "v2");

    std::cout<< "+ AFTER write_to_cassandra v2"<<std::endl;

    midict->sync();
    ///////////////////////////////////////////// 

    key = generateKey(0.5, 0);
    std::cout<< "+  key created"<< std::endl;
    cc_val = malloc(sizeof(int32_t));
    int v3_value=0x666;
    std::memcpy((char *)cc_val, &v3_value, sizeof(int32_t));
    w->write_to_cassandra((void*)key, (void*) cc_val, "v3");

    std::cout<< "+ AFTER write_to_cassandra v3"<<std::endl;

    midict->sync();
    ///////////////////////////////////////////// 
    free(cc_val);

    free(key);
    free(value);

}
