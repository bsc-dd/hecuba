#include <hecuba/StorageDict.h>
#include <hecuba/KeyClass.h>
#include <hecuba/ValueClass.h>
#include <iostream>
#include <hecuba/UUID.h>

using StringKeyClass = KeyClass <std::string,int32_t,int32_t>;
using StringMultipleValueClass = ValueClass<std::string,int32_t>;
class MultipleValueDict:public StorageDict <StringKeyClass,StringMultipleValueClass>{ };

using nestedValue = ValueClass <MultipleValueDict>;
class NestedDict:public StorageDict <StringKeyClass,nestedValue>{ };


void test_nested(HecubaSession& s) {
	MultipleValueDict mulD;
	std::cout<<"*  Instantiate mulD " << &mulD<<std::endl;

	s.registerObject(&mulD);
	std::string mulname("inner");
	mulD.make_persistent(mulname);
	std::cout<<"* "<<mulname<<" persisted mulD " << &mulD<<std::endl;


	StringMultipleValueClass mulv = StringMultipleValueClass("string", 42);
	StringKeyClass mulk = StringKeyClass("stringk", 43, 44);
	mulD[mulk]=mulv;

	mulD.sync();
	std::cout<<"* "<< mulname<<"  inserted values and synched" << "mulD: "<< &mulD << std::endl;


	NestedDict nd; // The constructor of Nested Dict intantiates a dummy key and a dummy value (in this case StorageDict) to extract info 
	std::cout<<"*   NestedDict nd instantiate " << &nd <<std::endl;

	s.registerObject(&nd);
	std::cout<<"*   NestedDict nd registered" << std::endl;
	nd.make_persistent("nestedDictionary");
	std::cout<<"* nestedDictionary persisted" << std::endl;

	nestedValue nv(mulD);
	std::cout<<"* nv(mulD) nestedValue instantiated and initialized with mulD" << std::endl;
	StringKeyClass strk = StringKeyClass("string key test",10,20);
	nd[strk] = nv;
	std::cout<<"* nestedDictionary insertion: n[strk] = nv" << std::endl;

	nd.sync();
	std::cout<<"* inserted values in nestedDicitionary andd synched" << std::endl;

	NestedDict ndread;
	s.registerObject(&ndread);
	ndread.getByAlias("nestedDictionary");

	nestedValue  mulDvalue = ndread[strk]; //retrieve the value: a dictionary
	std::cout<<"* mulDvalue = ndread[strk]" << std::endl;

	MultipleValueDict mulD_read = nestedValue::get<0>(mulDvalue); //extract the dictionary from the value class
	//std::string first_value = StringMultipleValueClass::get<0> (mulD_read[mulk]);
	StringMultipleValueClass tmp = mulD_read[mulk];
	std::string first_value = StringMultipleValueClass::get<0> (tmp);
	int32_t second_value = StringMultipleValueClass::get<1> (mulD_read[mulk]);
	if (first_value != "string") {
		std::cout << "Error extracting elements of the value of the inner dictionary. Got " << first_value << " should be 'string' "<< std::endl;
	}
	if (second_value != 42) {
		std::cout << "Error extracting elements of the value of the inner dictionary. Got " << second_value << " should be '42' "<< std::endl;
	}

}
int main() { 
	HecubaSession s; //connects with Cassandra
    test_nested(s);
   
	std::cout<<"END TEST" << std::endl;
}
