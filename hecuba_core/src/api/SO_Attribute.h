#ifndef __SO_ATTRIBUTE_H__
#define __SO_ATTRIBUTE_H__
#include "IStorage.h"
#include "StorageObject.h"


//class StorageObject; // forward declaration

template <class T>
class SO_Attribute {

public:
    friend class StorageObject;
    //SO_Attribute<T>(IStorage *so, std::string name){
    SO_Attribute<T>(StorageObject *so, const std::string& name){
        std::cout << "SO_Attribute mi extranyo constructor "<< std::endl;
        this->so = so;
        this->name = (char *) malloc(name.length() + 1);
        memcpy(this->name, name.c_str(), name.length() + 1);
        this->initialize = false;
        so->addAttrSpec(typeid(T).name(), name);
    }
    SO_Attribute() {
        std::cout << "SO_Attribute default constructor "<< std::endl;
    }

    SO_Attribute(T value) {
        this->attr_value = value;
        // if this case possible? to call to this constructor from the StorageObject constructor?
#if 0
        if (this->so != nullptr) {
            this->send_to_cassandra(value);
            initialize = true;
        }
#endif
    }


    SO_Attribute(SO_Attribute& attr_to_copy) {
        // if attr_to_copy has a so and it is not initialized we have to read it from cassandra. If has a so and it is initialized we copy the value from memory
        T value_to_copy;
        if (((attr_to_copy.so != nullptr) && (attr_to_copy.initialize ==true)) || (attr_to_copy.so == nullptr)) {
            value_to_copy = attr_to_copy.attr_value;
        } else {
            value_to_copy = attr_to_copy.read_from_cassandra();
        }
        if (so != nullptr) {
            this->send_to_cassandra(value_to_copy);
            initialize = true;
        } 
        this->attr_value = value_to_copy;
    }

    SO_Attribute &operator = (T value) {
        std::cout << "SO_Attribute: operator = value" << std::endl;
        this->attr_value = value;
        if (so != nullptr) {
            this->send_to_cassandra(value);
            initialize = true;
        }
    }
    SO_Attribute &operator = (SO_Attribute<T>& attr_to_copy) {
        std::cout << "SO_Attribute: operator = SO_Attribute" << std::endl;
        // if attr_to_copy has a so and it is not initialized we have to read it from cassandra. If has a so and it is initialized we copy the value from memory
        T value_to_copy;
        if ((attr_to_copy.so != nullptr and attr_to_copy.initialize ==true) || (attr_to_copy.so == nullptr)) {
            value_to_copy = attr_to_copy.attr_value;
        } else {
            value_to_copy = attr_to_copy.read_from_cassandra(); 
                                                               
        }
        if (so != nullptr) {
            this->send_to_cassandra(value_to_copy); 
            initialize = true;
        } 
        this->attr_value = value_to_copy;
    }

    operator T() {
        if ((so != nullptr) && (initialize ==false)) {
            attr_value = read_from_cassandra(); 
        } 
        return attr_value;
    };

    T read_from_cassandra() {
        std::string attr_name(name);
        so->getAttr(attr_name,(void *)&attr_value);
        return attr_value;
    }

    void send_to_cassandra(T value) {
        std::string attr_name(name);
        so->setAttr(attr_name,(void *)&value);
    }
    ~SO_Attribute() {
        std::cout << "SO_Attribute:: Soy el destructor y vengo a arruinarte el dia" << std::endl;
        if (name != nullptr) {
            free(name);
            name = nullptr;
        }
        so = nullptr;
    }

    private:
        //IStorage *so = nullptr;
        StorageObject *so = nullptr;
        bool initialize = false;
        //std::string name;//=std::string("NOT_DEFINED");
        char* name=nullptr;
        T attr_value;

        void setSO (IStorage* so){
            this->so = so;
        }
        void setInitialize (const bool init){
            this->initialize = init;
        }
        void setName (const std::string &name){
            char *tmp = (char*)malloc(name.length())+1;
            memcpy(tmp, name.c_str(),name.length()+1); 
            this->name = tmp;
        }

};
#endif /* __SO_ATTRIBUTE_H__ */
