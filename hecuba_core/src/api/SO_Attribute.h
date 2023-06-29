#ifndef __SO_ATTRIBUTE_H__
#define __SO_ATTRIBUTE_H__
#include "IStorage.h"
#include "Hecuba_StorageObject.h"
#include "debug.h"
#include "HecubaExtrae.h"


//class StorageObject; // forward declaration

template <class T>
class SO_Attribute {

public:
    //SO_Attribute<T>(IStorage *so, std::string name){
    SO_Attribute<T>(StorageObject *so, const std::string& name){
        HecubaExtrae_event(HECUBAEV, HECUBA_SO_ATTR|HECUBA_INSTANTIATION);
        DBG("SO_Attribute::constructor "<<this<< " with name ["<<name<<"] and StorageObj "<< so );
        this->so = so;
        this->name = name;
        so->addAttrSpec(typeid(T).name(), name);
        HecubaExtrae_event(HECUBAEV, HECUBA_END);
    }
    SO_Attribute() = delete;

    SO_Attribute(const T& value) {
        HecubaExtrae_event(HECUBAEV, HECUBA_SO_ATTR|HECUBA_INSTANTIATION);
        DBG( "SO_Attribute::constructor "<<this<< " with type "<< typeid(T).name() );
        this->attr_value = value;
        // if this case possible? to call to this constructor from the StorageObject constructor?
        HecubaExtrae_event(HECUBAEV, HECUBA_END);
    }

    SO_Attribute(SO_Attribute& attr_to_copy) {
        HecubaExtrae_event(HECUBAEV, HECUBA_SO_ATTR|HECUBA_INSTANTIATION);
        DBG( "SO_Attribute::copy constructor " << this << " from "<< &attr_to_copy );
        *this = attr_to_copy;
        HecubaExtrae_event(HECUBAEV, HECUBA_END);
    }

    /////////////////////////////////// MEGA TRICK BEGINs HERE ///////////////////////
    ////    If T is a IStorage we cannot specify the source value as const because it conflicts with the other version of the operator =,
    ////    where the source parameter is of type SO_Attribute<T> and cannot be const because the code modifies it
    ////    (when attr_value is not loaded in memory). If T is a basic type or a string it must be const (because the parameter is a
    ////    reference to avoid extra copies). So we need two different signatures: one with a const value and the other without the const.
    ////    We cannot use enable_if as second parameter or operator = because the interface only allows one parameter, for this reason we
    ////    have implemented the assignment functionality in a different function.
    ////    So we have 3 versions of the operator =: One for IStorage (const), one for the other types (no const) and one for
    ////    SO_Attribute<T>
    ////    The problem: this.inner_value = value;

    template <class V> void assignment (V& value, typename std::enable_if<std::is_base_of<IStorage,V>::value>::type* = 0){
        this->attr_value = value;
        if (so != nullptr) {
            this->send_to_cassandra(value);
        }
    }

    template <class V> void assignment (const V& value, typename std::enable_if<!std::is_base_of<IStorage,V>::value>::type* = 0){
        this->attr_value = value;
        if (so != nullptr) {
            this->send_to_cassandra(value);
        }
    }

    SO_Attribute &operator = (const T& value) {
        HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_WRITE);
        DBG("SO_Attribute: operator " << this << " = with const value on [" <<name<<"] " );
        assignment<T>(value);
        HecubaExtrae_event(HECUBAEV, HECUBA_END);
        return *this;
    }

    SO_Attribute &operator = (T& value) {
        HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_WRITE);
        DBG("SO_Attribute: operator " << this << " = value on [" <<name<<"] " );
        assignment<T>(value);
        HecubaExtrae_event(HECUBAEV, HECUBA_END);
        return *this;
    }

    SO_Attribute &operator = (SO_Attribute<T>& attr_to_copy) {
        DBG( "SO_Attribute: operator " << this << " = SO_Attribute " << &attr_to_copy );
        if (this == &attr_to_copy) return *this;
        // if attr_to_copy has a so and it is not initialized we have to read it from cassandra. Finally we copy the value from memory
        if (attr_to_copy.so != nullptr) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_READ);
            attr_to_copy.read_from_cassandra(); // read_from_cassanddra: if T is a IStorage attr_to_copy.attr_value will contain the instance of the IStorage. It's ok.
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }
        attr_value = attr_to_copy.attr_value;
        name = attr_to_copy.name;
        if (so != nullptr) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_WRITE);
            this->send_to_cassandra(attr_value);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        return *this;
    }
    ////////////////////////// MEGA TRICK ENDs HERE //////////

    operator T&() {
        DBG( "SO_Attribute::casting "<< name<< " to ["<< typeid(T).name() << "]");
        if (so != nullptr){
            HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_READ);
            read_from_cassandra();
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }
        return attr_value;
    };

    template <class V> void setAttributeValue(IStorage* sd, void* buffer, typename std::enable_if<std::is_base_of<IStorage, V>::value>::type* = 0 ) {
        uint64_t * uuid = *(uint64_t**) buffer;
        attr_value = instantiateIStorage<V>(sd, uuid);
        // if the value is of type IStorage is the case of nested IStorage and to avoid inconsistencies we always go to Cassandra to get the values
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
        v->setPersistence(uuid);
        sd->getCurrentSession().registerObject(v->getDataAccess(),v->getClassName());
        // enable stream
        if (v->isStream()){
            v->getObjSpec().enableStream();
            v->configureStream(UUID::UUID2str(uuid));
        }
        return *v;
    }

    /* Modifies 'this->attr_value' */
    void read_from_cassandra() {
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
        so->getAttr(name, buffer);
        this->template setAttributeValue<T>(this->so, buffer);
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
        so->setAttr(name,(void *)cast2IStorageBuffer(value)); // TRICK: We need to static_cast to IStorage to be able to find the base class due to the 'virtual'
    }
    ~SO_Attribute() {
        HecubaExtrae_event(HECUBAEV, HECUBA_SO_ATTR|HECUBA_DESTROY);
        DBG("SO_Attribute::destructor [" <<name<<"] "<< this );
        so = nullptr;
        HecubaExtrae_event(HECUBAEV, HECUBA_END);
    }

    private:
        //IStorage *so = nullptr;
        StorageObject *so = nullptr;
        std::string name = std::string("NOT_DEFINED");
        T attr_value;

        void setSO (IStorage* so){
            this->so = so;
        }

};
#endif /* __SO_ATTRIBUTE_H__ */
