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
    SO_Attribute() = delete;

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
        std::cout << "SO_Attribute::copy constructor " << this << " from "<< &attr_to_copy << std::endl;
        *this = attr_to_copy;
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
        std::cout << "SO_Attribute: operator " << this << " = SO_Attribute " << &attr_to_copy << std::endl;
        if (this == &attr_to_copy) return *this;
        // if attr_to_copy has a so and it is not initialized we have to read it from cassandra. Finally we copy the value from memory
        if (attr_to_copy.so != nullptr) {
            if (attr_to_copy.initialize == false) {
                attr_to_copy.read_from_cassandra(); // read_from_cassanddra: if T is a IStorage attr_to_copy.attr_value will contain the instance of the IStorage. It's ok.
            }
        }
        attr_value = attr_to_copy.attr_value;
        if (so != nullptr) {
            this->send_to_cassandra(attr_value);
        }
        initialize = true;
        name = attr_to_copy.name;
        return *this;
    }

    operator T() {
        if ((so != nullptr) && (initialize ==false)) {
            attr_value = read_from_cassandra(); 
        } 
        return attr_value;
    };

    template <class V> void setAttributeValue(IStorage* sd, void* buffer, typename std::enable_if<std::is_base_of<IStorage, V>::value>::type* = 0 ) {
        uint64_t * uuid = *(uint64_t**) buffer;
        attr_value = instantiateIStorage<V>(sd, uuid);
    }
    template <class V> void setAttributeValue(IStorage* sd, void* buffer, typename std::enable_if< std::is_base_of<std::string, V>::value >::type* = 0 ) {
        attr_value = std::string(*(char **)buffer);
    }
    template <class V> void setAttributeValue(IStorage* sd, void* buffer, typename std::enable_if<!std::is_base_of<IStorage, V>::value && !std::is_base_of<std::string, V>::value>::type* = 0 ) {
        attr_value = *(V*) buffer;
    }

    template <class V> V& instantiateIStorage(IStorage* sd,  uint64_t* uuid,
            typename std::enable_if<!std::is_base_of<IStorage, V>::value>::type* =0 ) {
        // DUMMY FUNCTION SFINAE enters into action
    }

    template <class V> V& instantiateIStorage(IStorage* sd,  uint64_t* uuid,
            typename std::enable_if<std::is_base_of<IStorage, V>::value>::type* =0 ) {
        V *v = new V();
        sd->getCurrentSession()->registerObject(v);
        v->setPersistence(uuid);
        // enable stream
        if (v->isStream()){
            v->getObjSpec().enableStream();
            v->configureStream(UUID::UUID2str(uuid));
        }
        return *v;
    }

    /* Modifies 'this->attr_value' */
    void read_from_cassandra() {
        std::string attr_name(name);
        // if T is a string or a IStorage: size = sizeof(char *)
        // if T is a basic type: size = sizeof(T)
        uint32_t attr_size;
        if ( (std::is_base_of<std::string, T>::value)
                || (std::is_base_of<IStorage, T>::value) ) {//ObjSpec::isBasicType(ObjSpec::c_to_cass(typeid(decltype(attr_value)).name()))) {
            attr_size = sizeof(char *);
        }else {
            attr_size = sizeof(T);
        }

        void* buffer = malloc(attr_size);
        so->getAttr(attr_name, buffer);
        this->template setAttributeValue<T>(this->so, buffer);
        initialize = true;
    }

    template <class V > char *cast2IStorageBuffer(const V& value,  typename std::enable_if<!std::is_base_of<IStorage, V>::value>::type* =0) {
		if (typeid(value) == typeid(std::string)){
			const std::string& valuestring = reinterpret_cast<const std::string&>(value);
            const char *valuetmp = valuestring.c_str();
            char *copyValue = (char *)malloc(strlen(valuetmp) + 1); // TODO: This is NEVER freed.
            strcpy(copyValue, valuetmp);
            int size = sizeof(copyValue);
            char *buf = (char *) malloc(size);
            memcpy(buf, &copyValue, size);
            return buf;
        } else {
            return (char*)&value;
        }
    }

    template <class V > char *cast2IStorageBuffer(const V& value,  typename std::enable_if<std::is_base_of<IStorage, V>::value>::type* =0) {
        int size = sizeof(V*);
        char * buf = (char*) malloc(size);
        IStorage* tmp = static_cast<IStorage*>(const_cast<V*>(&value));
        memcpy(buf, &tmp, size);
        return buf;
    }

    void send_to_cassandra(const T& value) {
        std::string attr_name(name);
        so->setAttr(attr_name,(void *)cast2IStorageBuffer(value)); // TRICK: We need to static_cast to IStorage to be able to find the base class due to the 'virtual'
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
