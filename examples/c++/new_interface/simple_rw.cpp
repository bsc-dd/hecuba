#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

#include <hecuba/StorageDict.h>
#include <hecuba/KeyClass.h>
#include <hecuba/ValueClass.h>
#define SIZE 3

using IntKeyClass = KeyClass<int32_t>;

using FloatValueClass = ValueClass<float>;


class MyDictClass: public StorageDict <IntKeyClass,FloatValueClass> {

};


void test_really_simple(HecubaSession &s,const char *name) {
    MyDictClass mydict;
    int tss[SIZE] ={42, 43, 44};
    float lats[SIZE]={0.666, 0.777, 0.888};

    s.registerObject(&mydict);
    mydict.make_persistent(name);




    mydict.sync();

//TODO ADD READ

    for (int i=0; i<SIZE; i++) {
	IntKeyClass k = IntKeyClass(tss[i]);
	FloatValueClass v;
        v = mydict[k];
	std::cout<< "iteration " << i<<": key "<< IntKeyClass::get<0>(k) << " value "<<FloatValueClass::get<0>(v)<<std::endl;
    }
   


}


int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;
    HecubaSession s;
    std::cout<< "+ Session started"<<std::endl;

    std::cout << "Starting test 1 " <<std::endl;
    test_really_simple(s,"mydict");


    std::cout << "End tests " <<std::endl;
}
