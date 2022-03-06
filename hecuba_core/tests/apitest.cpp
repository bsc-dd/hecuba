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

    //model_class: defines a StorageDict indexed by a float and an integer, with values of type numpy.ndarray
    s.loadDataModel("model_class.yaml","model_class.py");

    std::cout<< "+ Data Model loaded"<<std::endl;

    // create the dictionary:
    //      "midict" is the class name defined in model_class.yaml
    //      "outputDict" is the name of the persistent dictionary (the table in cassandra)

    IStorage* midict = s.createObject("midict", "outputDict");
    std::cout<< "+ 'dict' object created"<<std::endl;

    //generate the data to insert in the dictionary:
    //      key is a buffer that contains the two values of the key consecutive
    //      numpymeta is a buffer that contains: number of dimensions followed
    //                                           by the size of each dimension
    //      value is a buffer containing the numpy: just the float consecutive in C-order

    key = generateKey(0.5, 0);
    numpymeta = generateMetas();
    value = generateNumpyContent();

    // Create a new StorageNumpy initializing the value and the metas
    IStorage *mi_sn=s.createObject("hecuba.hnumpy.StorageNumpy","minp",numpymeta,value);

    mi_sn->sync();
    std::cout<< "Numpy creation completed"<<std::endl;

    // Add the numpy to the dictionary: midict[key] = mi_sn
    midict->setItem((void*)key, mi_sn);
    std::cout<< "Third setItem completed"<<std::endl;

    midict->sync();


    free(key);
    free(numpymeta);
    free(value);

}
