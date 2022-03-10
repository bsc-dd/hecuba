#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

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

    s.loadDataModel("model_storageobj.yaml","model_storageobj.py");
    std::cout<< "+ Data Model loaded"<<std::endl;

    // create a new StorageObject: "miclass" is the name of the class defined in model_storageobj.yaml 
    //                              and "miobj" is the name of the persistent object 

    IStorage* miobj = s.createObject("miclass", "miobj");
    std::cout<< "+ 'miobj' object created"<<std::endl;

    ///////////////////////////////////////////// 

    // set values to each attribute: name of the attribute (as specified in model_storageobj.yaml) followed by a pointer to the value

    // Attribute "lat" of type float
    float value = 0.666;
    miobj->setAttr("lat", &value);
    /////////////////////////////////////////////
    // Attribute "ts" of type int
    int value2 = 42;
    miobj->setAttr("ts", &value2);
    ///////////////////////////////////////////// 
    // Attribute "minp" of type numpy: we first create the StorageNumpy passing
    // the values and the numpy metas and then set the attribute in the
    // StorageObj
    char * numpymeta;
    numpymeta = generateMetas();
    char *valueNP = generateNumpyContent();
    IStorage *mi_sn=s.createObject("hecuba.hnumpy.StorageNumpy","minp",numpymeta,valueNP);
    miobj->setAttr("minp",mi_sn);
    ///////////////////////////////////////////// 
    // Attribute miso of type miclass: we first create the StorageObject miobj2
    // and then we can set the attribute of miobj, initialize all the
    // attributes and then set the attribute in the StorageObj
    IStorage* miobj2 = s.createObject("miclass", "miobj2");
    miobj2->setAttr("lat", &value);
    miobj2->setAttr("ts", &value2);
    miobj2->setAttr("minp",mi_sn);

    miobj->setAttr("miso", miobj2);
    ///////////////////////////////////////////// 
    IStorage* midict = s.createObject("midict", "outputDict");
    int key = 42;
    mi_sn=s.createObject("hecuba.hnumpy.StorageNumpy","otronp",numpymeta,valueNP);
    midict->setItem(&key, mi_sn);
    midict->sync();

    miobj->sync();
}
