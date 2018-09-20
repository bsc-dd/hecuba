#include <string>

#ifndef HFETCH_MODULEEXCEPTION_H
#define HFETCH_MODULEEXCEPTION_H


class ModuleException : public std::exception {
public:

    ModuleException() {};

    explicit ModuleException(const std::string &message);

    virtual ~ModuleException() throw() {}

    virtual const char *what() const throw() {
        return exc_msg.c_str();
    }

protected:

    std::string exc_msg;
};


class TypeErrorException : public ModuleException {
public:

    TypeErrorException() {};

    explicit TypeErrorException(const std::string &message);

};

#endif //HFETCH_MODULEEXCEPTION_H
