#include <map>
#include <iostream>
#include <type_traits>
#include <cxxabi.h>
#include <typeindex>
#include <vector>
#include <string>


#define EVAL(...) EVAL1024(__VA_ARGS__)
#define EVAL1024(...) EVAL512(EVAL512(__VA_ARGS__))
#define EVAL512(...) EVAL256(EVAL256(__VA_ARGS__))
#define EVAL256(...) EVAL128(EVAL128(__VA_ARGS__))
#define EVAL128(...) EVAL64(EVAL64(__VA_ARGS__))
#define EVAL64(...) EVAL32(EVAL32(__VA_ARGS__))
#define EVAL32(...) EVAL16(EVAL16(__VA_ARGS__))
#define EVAL16(...) EVAL8(EVAL8(__VA_ARGS__))
#define EVAL8(...) EVAL4(EVAL4(__VA_ARGS__))
#define EVAL4(...) EVAL2(EVAL2(__VA_ARGS__))
#define EVAL2(...) EVAL1(EVAL1(__VA_ARGS__))
#define EVAL1(...) __VA_ARGS__

#define EMPTY()
#define DEFER1(m) m EMPTY()
#define DEFER2(m) m EMPTY EMPTY()()

#define FIRST(a, ...) a
#define SECOND(a, b, ...) b

#define IS_PROBE(...) SECOND(__VA_ARGS__, 0)
#define PROBE() ~, 1

#define CAT(a,b) a ## b

#define NOT(x) IS_PROBE(CAT(_NOT_,x))
#define _NOT_0 PROBE()

#define BOOL(x) NOT(NOT(x))

#define HAS_ARGS(...) BOOL(FIRST(_END_OF_ARGUMENTS_ __VA_ARGS__)())
#define _END_OF_ARGUMENTS_() 0

#define IF_ELSE(condition) _IF_ELSE(BOOL(condition))
#define _IF_ELSE(condition) CAT(_IF_, condition)

#define _IF_1(...) __VA_ARGS__ _IF_1_ELSE
#define _IF_0(...)             _IF_0_ELSE

#define _IF_1_ELSE(...)
#define _IF_0_ELSE(...) __VA_ARGS__

#define DECL(t1,n1, ...) \
    SO_Attribute<t1> n1={this,#n1}; \
    IF_ELSE(HAS_ARGS(__VA_ARGS__))( \
           DEFER2(_DECL)()(__VA_ARGS__)\
    ) ( )
#define _DECL() DECL

#define HECUBA_ATTRS(t1,n1,...) SO_ClassName myname = {this, typeid(*this).name()}; EVAL(DECL(t1,n1,__VA_ARGS__))
//
//class miso: public StorageObject {
//  public:
//      HECUBA_ATTRS(int,a,float,b)
//};
//
//This expands to:
//class miso:publicStorageObject
//class innerSO: public StorageObject {
//    public:
//        SO_ClassName myname  ={this, typeid(*this).name()};
//        SO_Attribute<int>   a={this, "a"};
//        SO_Attribute<float> b={this, "b"};
//};
//

//recursive behaviour extracted from http://jhnet.co.uk/articles/cpp_magic

