#include "SO_ClassName.h"
SO_ClassName::SO_ClassName (StorageObject* so, const std::string& name) {
        int32_t status;
        so->setClassName(abi::__cxa_demangle(name.c_str(),NULL,NULL,&status));
    }
