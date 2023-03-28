#ifndef __HECUBA_SO_CLASSNAME_H__
#define __HECUBA_SO_CLASSNAME_H__
#include <typeinfo>
#include <cxxabi.h>
#include <string>
#include "Hecuba_StorageObject.h"

class StorageObject; // forward declaration
class SO_ClassName{
    public:
    SO_ClassName(StorageObject* so, const std::string& name);
}; 

#endif /* __HECUBA_SO_CLASSNAME_H__ */
