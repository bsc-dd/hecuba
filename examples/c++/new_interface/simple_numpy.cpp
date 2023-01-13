#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

#include <hecuba/StorageNumpy.h>
#include <hecuba/KeyClass.h>
#include <hecuba/ValueClass.h>
#define COLS 3
#define ROWS 4


char * generateNumpyContent(const std::vector<uint32_t> &metas) {

    double *numpy=(double*)malloc(sizeof(double)*metas[0]*metas[1]);
    double *tmp = numpy;
    double num = 1;
    for (int i=0; i<metas[1]; i++) {
        for (int j=0; j<metas[0]; j++) {
            std::cout<< "++ "<<i<<","<<j<<std::endl;
            *tmp = num++;
            tmp+=1;
        }
    }
    return (char*) numpy;
}

void test_really_simple(HecubaSession &s,const char *name) {
    std::vector<uint32_t> metadata = {3, 4};
    char *data = generateNumpyContent(metadata);
    StorageNumpy sn(data,metadata);

    s.registerObject(&sn);
    sn.make_persistent(name);

    sn.sync();
}



int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;
    HecubaSession s;
    std::cout<< "+ Session started"<<std::endl;

    std::cout << "Starting test 1 " <<std::endl;
    test_really_simple(s,"mynumpy");

    std::cout << "End tests " <<std::endl;
}
