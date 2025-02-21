
#include <StorageNumpy.h>
#include <StorageStream.h>
#include <iostream>

class mynumpyclass: public StorageNumpy, public StorageStream {

};

void consumer_singleNumpy_int() {
mynumpyclass mysn;

mysn.getByAlias("myintnumpy");
mysn.poll(); //wait on the stream for the numpy

uint64_t *p = (uint64_t *) mysn.data;
std::vector<unsigned int> metas = mysn.metas;

for (int i=0; i<metas[0]; i++) {
    for(int j=0; j<metas[1]; j++) {
        std::cout << *p << std::endl;
        p++;
    }
}

mynumpyclass mysn2;
mysn2.getByAlias("myintnumpy");
mysn2.setNumpy(); //read the numpy from Cassandra

p = (uint64_t*) mysn2.data;
metas = mysn2.metas;

for (int i=0; i<metas[0]; i++) {
    for(int j=0; j<metas[1]; j++) {
        std::cout << *p << std::endl;
        p++;
    }
}


}

void consumer_singleNumpy_float() {
mynumpyclass mysn;

mysn.getByAlias("myfloatnumpy");
mysn.poll(); //wait on the stream for the numpy

double *p = (double *) mysn.data;
std::vector<unsigned int> metas = mysn.metas;

for (int i=0; i<metas[0]; i++) {
    for(int j=0; j<metas[1]; j++) {
        std::cout << *p << std::endl;
        p++;
    }
}

mynumpyclass mysn2;
mysn2.getByAlias("myfloatnumpy");
mysn2.setNumpy(); //read the numpy from Cassandra

p = (double*) mysn2.data;
metas = mysn2.metas;

for (int i=0; i<metas[0]; i++) {
    for(int j=0; j<metas[1]; j++) {
        std::cout << *p << std::endl;
        p++;
    }
}


}

main () {
    consumer_singleNumpy_int();
    consumer_singleNumpy_float();
}
