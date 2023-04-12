#include <iostream>

#include <StorageDict.h>
#include <KeyClass.h>
#include <ValueClass.h>
#define SIZE 3

using IntKeyClass = KeyClass<int32_t>;

using FloatValueClass = ValueClass<float>;

class MyDictClass: public StorageDict <IntKeyClass,FloatValueClass,MyDictClass> {

};


void test_really_simple(const char *name) {
	MyDictClass mydict;
	int tss[SIZE] ={42, 43, 44};
	float lats[SIZE]={0.666, 0.777, 0.888};

	mydict.make_persistent(name);
	for (int i=0; i<SIZE; i++) {
		IntKeyClass k = IntKeyClass(tss[i]);
		FloatValueClass v = FloatValueClass(lats[i]);
		mydict[k]=v;
	}

	mydict.sync();

	std::cout << "Insertion Completed" << std::endl;

	MyDictClass mydict_read;
	mydict_read.getByAlias(name);	
	std::cout << "Starting object read" << std::endl;

	for (int i=0; i<SIZE; i++) {
		IntKeyClass k = IntKeyClass(tss[i]);
		FloatValueClass v;
		v = mydict_read[k];
		if (FloatValueClass::get<0>(v) != lats[i])  {
			std::cout << "Value read differs from value inserted " << std::endl;
			std::cout<< "Iteration " << i <<": key "<< IntKeyClass::get<0>(k) << " value "<<FloatValueClass::get<0>(v)<< " should be " << lats[i] << std::endl;
		}
	}
   
}


int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;

    std::cout << "Starting test 1 " <<std::endl;
    test_really_simple("mydict");


    std::cout << "End tests " <<std::endl;
}
