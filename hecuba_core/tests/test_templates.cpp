#include <hecuba/StorageDict.h>
#include <hecuba/KeyClass.h>
#include <hecuba/ValueClass.h>
#include <iostream>

using MyKeyClass = KeyClass <float,int32_t,int32_t>;
using MyValueClass = ValueClass<int32_t>;
class MyDict:public StorageDict <MyKeyClass,MyValueClass>{
};

using StringKeyClass = KeyClass <std::string,int32_t,int32_t>;
using StringValueClass = ValueClass<std::string>;
class StringDict:public StorageDict <StringKeyClass,StringValueClass>{
};

using StringMultipleValueClass = ValueClass<std::string,int32_t>;
class MultipleValueDict:public StorageDict <StringKeyClass,StringMultipleValueClass>{
};

int main() {

HecubaSession s; //connects with Cassandra
MyDict d; 	//Mydict d(name) --> this does not work if the user do not add the explicit class to its defined class 
		//the make persistent method cannot be invoked until the object is registered. 
		//can be implicitily called by registerObject if name is set

s.registerObject(&d); 		// Generates the python file with the class definition. 
		      		// if pending to be persistent we do it here
std::string name("d_mydict");
d.make_persistent(name); 	//requires to be registered. Returns error if it is not registered

MyKeyClass k = MyKeyClass((float)3.5,666,2);
MyValueClass v = MyValueClass(666);

d[k]=v; 			//if pending to be persisted returns error


StringDict strD;
s.registerObject(&strD);
std::string strname("strd_name");
strD.make_persistent(strname);

StringKeyClass strk = StringKeyClass("string key test",10,20);
StringValueClass strv = StringValueClass("string value test");

strD[strk]=strv;

MultipleValueDict mulD;
s.registerObject(&mulD);
std::string mulname("mul_name");
mulD.make_persistent(mulname);
StringMultipleValueClass mulv = StringMultipleValueClass("string", 42);
StringKeyClass mulk = StringKeyClass("stringk", 43, 44);

mulD[mulk]=mulv;

}
