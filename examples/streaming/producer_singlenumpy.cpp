
#include <StorageNumpy.h>
#include <StorageStream.h>
#include <iostream>

#define ROWS 3
#define COLS 4

class mynumpyclass: public StorageNumpy, public StorageStream {

};

char * generateNumpyContent_int(const std::vector<uint32_t> &metas) {

    uint64_t *numpy=(uint64_t *)malloc(sizeof(uint64_t )*metas[0]*metas[1]);
    uint64_t  *tmp = numpy;
    uint64_t  num = 1;
    for (int i=0; i<metas[0]; i++) {
        for (int j=0; j<metas[1]; j++) {
            *tmp = num++;
            std::cout<< "++ "<<i<<","<<j<<":" << *tmp<< std::endl;
            tmp+=1;
        }
    }
    std::cout<< "+ Generated NUMPY ["<<metas[0]<<", "<<metas[1]<<"] using "<<sizeof(int)*metas[0]*metas[1]<<"bytes at "<<std::hex<<(void*)numpy<<std::endl;
    return (char*) numpy;
}

void producer_singleNumpy_int() {
mynumpyclass mysn;
std::vector<uint32_t> metadata = {ROWS, COLS};
char* data = generateNumpyContent_int(metadata);

    mysn.setNumpy(data,metadata,'i');
	mysn.make_persistent("myintnumpy");
    mysn.sync();
    mysn.send();

}
char * generateNumpyContent_float(const std::vector<uint32_t> &metas) {

    double *numpy=(double *)malloc(sizeof(double)*metas[0]*metas[1]);
    double  *tmp = numpy;
    double  num = 1.0;
    for (int i=0; i<metas[0]; i++) {
        for (int j=0; j<metas[1]; j++) {
            *tmp = num++;
            std::cout<< "++ "<<i<<","<<j<<":" << *tmp<< std::endl;
            tmp+=1;
        }
    }
    std::cout<< "+ Generated NUMPY ["<<metas[0]<<", "<<metas[1]<<"] using "<<sizeof(double)*metas[0]*metas[1]<<"bytes at "<<std::hex<<(void*)numpy<<std::endl;
    return (char*) numpy;
}

void producer_singleNumpy_float() {
mynumpyclass mysn;
std::vector<uint32_t> metadata = {ROWS, COLS};
char* data = generateNumpyContent_float(metadata);

    mysn.setNumpy(data,metadata,'f');
	mysn.make_persistent("myfloatnumpy");
    mysn.sync();
    mysn.send();

}

main () {
    producer_singleNumpy_int();
    producer_singleNumpy_float();
}
