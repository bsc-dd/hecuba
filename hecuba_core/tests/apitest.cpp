#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

char * generateKey(double lat, int ts) {

    char * key = (char*) malloc (sizeof(double) + sizeof(int));
    double *lat_key = (double*) key;
    *lat_key = lat;
    int *ts_key = (int*) (key + sizeof(double));
    *ts_key = ts;
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

    s.loadDataModel("DUMMYMODELFILE.MDL");
    std::cout<< "+ Data Model loaded"<<std::endl;

    IStorage* midict = s.createObject("dataModel", "yoli");
    std::cout<< "+ 'dict' object created"<<std::endl;

    key = generateKey(0.5, 0);
    std::cout<< "+  key created"<<std::endl;
    numpymeta = generateMetas();
    std::cout<< "+ metadata created"<<std::endl;
    value = generateNumpyContent();
    std::cout<< "+ value created at "<<std::hex<<(void*)value<<std::endl;

    // midict[key] = value;
    midict->setItem((void*)key, (void*) value, NULL, (void*) numpymeta);

    free(key);
    free(numpymeta);
    free(value);

}
