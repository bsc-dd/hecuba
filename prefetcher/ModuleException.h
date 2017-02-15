#include <string>

#ifndef HFETCH_MODULEEXCEPTION_H
#define HFETCH_MODULEEXCEPTION_H


class ModuleException :public std::exception {
public:

    explicit ModuleException(const std::string& message);


        virtual const char* what() const throw()
        {
            return exc_msg.c_str();
        }

private:

    std::string exc_msg;
};


#endif //HFETCH_MODULEEXCEPTION_H
