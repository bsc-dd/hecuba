#include "ModuleException.h"


ModuleException::ModuleException(const std::string &message) : exc_msg(message) {
    //Logging action
}

TypeErrorException::TypeErrorException(const std::string &message) : ModuleException(message) {
    //Type Error action
}