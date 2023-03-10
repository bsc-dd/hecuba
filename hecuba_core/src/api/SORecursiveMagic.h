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

//#define MAP(m, first,second, ...)           \
        m(first,second)                           \
        IF_ELSE(HAS_ARGS(__VA_ARGS__))(    \
              DEFER2(_MAP)()(m, __VA_ARGS__)   \
        )(                                 \
                    /* Do nothing, just terminate */ \
        )

//#define MAP_1par(m, first, ...)           \
        m(first)                           \
        IF_ELSE(HAS_ARGS(__VA_ARGS__))(    \
              DEFER2(_MAP_1par)()(m, __VA_ARGS__)   \
        )(                                 \
                    /* Do nothing, just terminate */ \
        )
//#define _MAP_1par() MAP_1par

// #define _MAP() MAP

//#define TOFIELD(t1,n1) FIELD(t1,n1)

//#define FIELD(t1,n1) t1 n1;

#define VECTOR(t1,n1, ...) \
    {typeid(t1).name(), std::string(#n1)} \
    IF_ELSE(HAS_ARGS(__VA_ARGS__))( \
            , DEFER2(_VECTOR)()(__VA_ARGS__)\
    ) ( )

#define _VECTOR() VECTOR

//#define DECLARE(t1,n1,...) \
    EVAL(MAP(FIELD,t1,n1,__VA_ARGS__)) \
    std::vector<std::pair<std::string,std::string>>c_types_attr={ EVAL(VECTOR(t1,n1,__VA_ARGS__)) };\
    std::vector<std::pair<std::string, std::string>> get_attributes_info() {return c_types_attr;}; 
    
#if 0
#define GEN_STRUCT(t1,n1, ...) \
    &n1 \
    IF_ELSE(HAS_ARGS(__VA_ARGS__))( \
          , DEFER2(_GEN_STRUCT)()(__VA_ARGS__)\
    ) ( )
#define _GEN_STRUCT() GEN_STRUCT
#endif

#define DECL(t1,n1, ...) \
    SO_Attribute<t1> n1={this,#n1}; \
    IF_ELSE(HAS_ARGS(__VA_ARGS__))( \
           DEFER2(_DECL)()(__VA_ARGS__)\
    ) ( )
#define _DECL() DECL 

    // it is not necessary to pass the address of the attributes
    //#define DEFAULT_CONSTRUCTOR(clase, t1,n1,...)  EVAL(DECL(t1,n1,__VA_ARGS__)) clase():StorageObject({ EVAL(VECTOR(t1,n1,__VA_ARGS__))},{EVAL(GEN_STRUCT(t1,n1,__VA_ARGS__))})

#define DEFAULT_CONSTRUCTOR(clase, t1,n1,...)  EVAL(DECL(t1,n1,__VA_ARGS__)) clase():StorageObject({ EVAL(VECTOR(t1,n1,__VA_ARGS__))})

// it is not necessary to pass the address of the attributes
//#define SO_CONSTRUCTOR(t1,n1,...) StorageObject({ EVAL(VECTOR(t1,n1,__VA_ARGS__))},{EVAL(GEN_STRUCT(t1,n1,__VA_ARGS__))})
#define SO_CONSTRUCTOR(t1,n1,...) StorageObject({ EVAL(VECTOR(t1,n1,__VA_ARGS__))})

//this should be used to declare the attributes if the SO_CONSTRUCTOR is used and DEFAULT_CONSTRUCTOR is not used
#define HECUBA_ATTRS(t1,n1,...) EVAL(DECL(t1,n1,__VA_ARGS__)) 
//miso: public StorageObject {
//      DEFAULT_CONSTRUCTOR(miso,int,a,float,b) {
//          
//      }
//      miso(par1,par2): SO_CONSTRUCTOR(int,a,float,b){
//      }
//}
//
//miso: public StorageObject {
//      DECLARE_ATTRS(int,a,float,b)
//      miso(par1,par2): SO_CONSTRUCTOR(int,a,float,b) {
//
//      }
//}
//
//Now it works both with constructor or just using DECLARE_ATTRS

//recursive behaviour extracted from http://jhnet.co.uk/articles/cpp_magic

