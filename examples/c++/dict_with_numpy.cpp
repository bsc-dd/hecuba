#include <StorageDict.h>
#include <StorageNumpy.h>
#include <KeyClass.h>
#include <ValueClass.h>
#include <iostream>
#include <UUID.h>

#define COLS 3
#define ROWS 4

using IntKeyClass = KeyClass <int32_t>;

using nestedValue = ValueClass <StorageNumpy>;
class NestedDict:public StorageDict <IntKeyClass,nestedValue,NestedDict>{ };



double * generateNumpyContent(const std::vector<uint32_t> &metas) {

    double *numpy=(double*)malloc(sizeof(double)*metas[0]*metas[1]);
    double *tmp = numpy;
    double num = 0;
    for (int i=0; i<metas[0]; i++) {
        *tmp = (num++ );
        //std::cout<< "++ "<<i<< " ==> " << (*tmp) << std::endl;
        tmp+=1;
    }
    return (double*) numpy;
}

bool equalsNumpy(const StorageNumpy& s, double *data, std::vector<uint32_t> metas) {
    // Check metas
    if (s.metas.size() != metas.size()) {
        std::cout<< "++ dimensions differs! "<< s.metas.size()<< " != " << metas.size() << std::endl;
        return false;
    }
    for (int i=0; i<metas.size(); i++) {
        if (s.metas[i] != metas[i]) {
            std::cout<< "++ dimension "<<i<<" differs! "<< s.metas[i]<< " != " << metas[i] << std::endl;
            return false;
        }
    }
    // Check Content
    double* p = (double*)s.data;
    double* q = data;
    for (int i=0; i<metas[0]; i++) {
            if (*p != *q) {
                return false;
            }
            p++;
            q++;
    }
    return true;
}


void test_dict_with_numpy() {
    std::vector<uint32_t> metadata = {ROWS, COLS};
    double *data = generateNumpyContent(metadata);
    StorageNumpy sn(data,metadata);

    sn.make_persistent("inner");

    sn.sync();

	NestedDict nd; // The constructor of Nested Dict intantiates a dummy key and a dummy value (in this case StorageDict) to extract info 
	std::cout<<"*   NestedDict nd instantiate " << &nd <<std::endl;

	nd.make_persistent("Dictionary");
	std::cout<<"* nestedDictionary persisted" << std::endl;

	nestedValue nv(sn);
	std::cout<<"* nv(sn) nestedValue instantiated and initialized with sn" << std::endl;
	IntKeyClass index = IntKeyClass(42);
	nd[index] = nv;
	std::cout<<"* nestedDictionary insertion: n[index] = nv" << std::endl;

	nd.sync();
	std::cout<<"* inserted values in nestedDicitionary andd synched" << std::endl;

	NestedDict ndread;
	ndread.getByAlias("Dictionary");

	// TODO: this does not work: nestedValue  snValue = ndread[index]. We need to review the differences between the copy constructor and the assignment
	nestedValue  snValue;
        snValue	= ndread[index]; //retrieve the value: a StorageNumpy
	std::cout<<"* snValue = ndread[index]" << std::endl;

	StorageNumpy  sn_read = nestedValue::get<0>(snValue); //extract the dictionary from the value class
	//std::string first_value = StringMultipleValueClass::get<0> (sn_read[mulk]);
	if (equalsNumpy(sn_read, data, metadata)) {
		std::cout << "OK: retrieved StorageNumpy matches stored StorageNumpy " << std::endl;
	} else {
		std::cout << "KO: retrieved StorageNumpy differs from stored StorageNumpy " << std::endl;

	}

}
int main() { 
    test_dict_with_numpy();
   
	std::cout<<"END TEST" << std::endl;
}
