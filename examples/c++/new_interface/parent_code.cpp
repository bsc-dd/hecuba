#include <iostream>
#include <hecuba/HecubaSession.h>
#include <hecuba/StorageNumpy.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>


char * generateNumpyContent(const std::vector<uint32_t> &metas) {

    double *numpy=(double*)malloc( sizeof(double) * metas[0] * metas[1]);
    double *tmp = numpy;
            *tmp =1.0;
    double num = 1;
    for (int i=0; i<metas[0]; i++) {
        for (int j=0; j<metas[1]; j++) {
            *tmp = num++;
            std::cout<< "++ "<<i<<","<<j<< "=" << *tmp << std::endl;
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

void check_and_deallocate(const std::string& numpyname, char* data, const std::vector<uint32_t>& metas) {
    StorageNumpy snread;
    bool found = false;
    while (!found) {
        try{
            snread.getByAlias(numpyname); // try to load the numpy, generates an exception if no data is found in the DB
            if (!equalsNumpy(snread, (double*)data, metas)) {
                std::cout << "Retrieved Numpy ["<< numpyname<< "] contains unexpected content (differnt from stored). " <<std::endl;
            } else {
                std::cout << "TEST check_and_deallocate PASSED"<<std::endl;
            }
            found = true;
        } catch (std::exception &e) {
            std::cout<< "StorageNumpy is not stored: " << std::string (e.what()) << std::endl;
        }
    }
}

int main(int argc, char**argv) {
    std::cout<< " --> Enter main " << std::endl;
    std::string numpy_name_noshared(argv[1]);

    std::vector<uint32_t> metas = {3,4};
    char * data= generateNumpyContent(metas);
    check_and_deallocate(numpy_name_noshared, data, metas);
    std::cout<< " <-- Exit main " << std::endl;
}
