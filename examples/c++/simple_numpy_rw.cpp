#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

#include <hecuba/StorageNumpy.h>

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

bool equalsNumpy(const StorageNumpy& s, double *data, std::vector<uint32_t> metas) {
	// Check metas
	if (s.metas.size() != metas.size()) {
		return false;
	}
	for (int i=0; i<metas.size(); i++) {
		if (s.metas[i] != metas[i]) {
			return false;
		}
	}
	// Check Content
	double* p = (double*)s.data;
	double* q = data;
	for (int i=0; i<metas[0]; i++) {
		for (int j=0; j<metas[1]; j++) {
			if (*p != *q) {
				std::cout<< "++ "<<i<<","<<j<< "=" << *p << " == " << *q << std::endl;
				return false;
			}
			p++;
			q++;
		}
	}
	return true;
}


void test_retrieve_simple(const char *name) {
    std::vector<uint32_t> metadata = {3, 4};
    char *data = generateNumpyContent(metadata);
    StorageNumpy sn(data,metadata);

    sn.make_persistent(name);

    sn.sync();

    StorageNumpy sn2;
    sn2.getByAlias(name);
 
    if (!equalsNumpy(sn2, (double*)data, metadata)) {
    	std::cout << "Retrieved Numpy ["<< name<< "] contains unexpected content (differnt from stored). " <<std::endl;
    }
}



int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;

    std::cout << "Starting test 1 " <<std::endl;
    test_retrieve_simple("mynumpytoread");

    std::cout << "End tests " <<std::endl;
}
