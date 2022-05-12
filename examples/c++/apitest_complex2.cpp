#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

#define ROWS 4
#define COLS 3
char * generateKey(char* lat) {

    char * key = (char*) malloc (sizeof(char*));
    char * keycontent = (char*) malloc (strlen(lat)+1);
    memcpy(keycontent, lat, strlen(lat)+1);
    memcpy(key, &keycontent,sizeof(char*));
    return key;
}

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

    s.loadDataModel("new_model.yaml","new_model.py");
    std::cout<< "+ Data Model loaded"<<std::endl;

    // create a new StorageObject: "myclass" is the name of the class defined in model_complex.yaml and "mysim" is the name of the persistent object 
    IStorage* mydict = s.createObject("mydict", "outputDict");

	char* lat = "hey";
    char *key;
    key=generateKey(lat);
    char * numpymeta;
    numpymeta = generateMetas();
    char *valueNP = generateNumpyContent();
    IStorage *mi_sn=s.createObject("hecuba.hnumpy.StorageNumpy","minp",numpymeta,valueNP);


    mydict->setItem(key, mi_sn);


    std::cout<< "+ completed: syncing"<<std::endl;

    // we sync every thing before ending the process
#if 0
    mi_sn->sync();
    mydict->sync();
    myobj->sync();
#endif
}
