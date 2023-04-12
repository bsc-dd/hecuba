#include <iostream>

#include <StorageNumpy.h>
#define COLS 3
#define ROWS 4


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

void test_really_simple(const char *name) {
    std::vector<uint32_t> metadata = {3, 4};
    char *data = generateNumpyContent(metadata);
    StorageNumpy sn(data,metadata);

    sn.make_persistent(name);

    sn.sync();
}



int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;

    std::cout << "Starting test 1 " <<std::endl;
    test_really_simple("mynumpy");

    std::cout << "End tests " <<std::endl;
}
