#ifndef _ATTR_CLASS_
#define _ATTR_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>
#include <tuple>
#include <type_traits>

#include "debug.h"
#include "IStorage.h"
#include "StorageDict.h"
#include "UUID.h"



template <class V1, class...rest>
class AttributeClass {

    public:
        AttributeClass() =default; 

        AttributeClass(const AttributeClass& a) {
            *this = a;
        }

        ~AttributeClass() {
            HecubaExtrae_event(HECUBAEV, HECUBA_ATTRCLASS|HECUBA_DESTROY);
            if (valuesBuffer != nullptr) {
                free(valuesBuffer);
                valuesBuffer = nullptr;
            }
            if (pendingKeysBuffer != nullptr) {
                free(pendingKeysBuffer);
                pendingKeysBuffer = nullptr;
            }
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        };
        // Constructor called when instantiating a new value with parameters: MyValueClass v(value);
        AttributeClass(std::string attrBaseName, const V1& part, rest... vals) {
            HecubaExtrae_event(HECUBAEV, HECUBA_ATTRCLASS|HECUBA_INSTANTIATION);
            //DEBUG("Number of clustering keys: "<<sizeof...(values)<<std::endl);
            // first element is the partition key, the rest of elements are clustering keys
            // only basic classes are suported
            manageAttr<V1>(attrBaseName,part);
            manageRest(attrBaseName,vals...);
            createValuesBuffer();
            values=std::make_tuple(part, vals...);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        AttributeClass& operator = (const AttributeClass& a) {
            HecubaExtrae_event(HECUBAEV, HECUBA_ATTRCLASS|HECUBA_ASSIGNMENT);
            std::cout << "Copy constructor Attribute Class" << std::endl;
            valuesDesc = a.valuesDesc;
            managedValues = a.managedValues;
            total_size = a.total_size;
            sd = a.sd;
            pendingKeysBuffer = a.pendingKeysBuffer;
            pendingKeysBufferSize = a.pendingKeysBufferSize;
            if (a.valuesBuffer != nullptr) {
                valuesBuffer = (char *) malloc(a.total_size);
                memcpy(valuesBuffer, a.valuesBuffer, a.total_size);
            } else {
                valuesBuffer = nullptr;
            }	
            values = a.values;

            HecubaExtrae_event(HECUBAEV, HECUBA_END);
            return *this;
        }

        template <class V> void manageAttr(std::string attrBaseName, const V& value) {
            DBG("Clust Key " <<std::string(typeid(decltype(value)).name())<<std::endl);
            std::string valuetype=ObjSpec::c_to_cass(typeid(decltype(value)).name());

            std::pair<std::string, std::string> valuedesc(attrBaseName+std::to_string(managedValues), valuetype);
            valuesDesc.push_back(valuedesc);
            createAttributeBuffer <V>(valuetype,value);
            managedValues++;
        }

        template <class V1alt, class...restalt> void manageRest(std::string attrBaseName, const V1alt& part, restalt... restValues){
            /* deal with the first parameter */
            DBG("Clust Key " << part << " "<<std::string(typeid(decltype(part)).name())<<std::endl);
            std::string valuetype=ObjSpec::c_to_cass(typeid(decltype(part)).name());
            std::pair<std::string, std::string> valuedesc(attrBaseName+std::to_string(managedValues), valuetype);
            valuesDesc.push_back(valuedesc);
            createAttributeBuffer <V1alt>(valuetype,part);
            managedValues++;
            manageRest(attrBaseName, restValues...);
        }

        void manageRest(std::string attrBaseName) {
        }

        template < class V1alt> V1alt& instantiateIStorage(IStorage* sd,  uint64_t* uuid,
                typename std::enable_if<!std::is_base_of<IStorage, V1alt>::value>::type* =0 ) {
            // DUMMY FUNCTION SFINAE enters into action
        }

        template < class V1alt> V1alt& instantiateIStorage(IStorage* sd,  uint64_t* uuid,
                typename std::enable_if<std::is_base_of<IStorage, V1alt>::value>::type* =0 ) {
            V1alt *v = new V1alt();
            //sd->getCurrentSession().registerObject(v); move to setPersistence and object constructor
            v->setPersistence(uuid);
            // enable stream
            if (v->isStream()){
                v->getObjSpec().enableStream();
                v->configureStream(UUID::UUID2str(uuid));
            }

            return *v;
        }

        // NOTE: we need the hecuba session, therefore pass through the IStorage containing it (sd)
        template <std::size_t ix, class V1alt, class...restalt> void setTupleValues(IStorage* sd, void *buffer) {

            size_t tam = 0;
            if (ObjSpec::isBasicType(valuesDesc[ix].second) ) {
                if (valuesDesc[ix].second == "text") {
                    (std::string &&)std::get<ix>(values)=std::string(*(char **)buffer);
                    tam = sizeof(char *);
                } else {
                    std::get<ix>(values)=*(V1alt *)buffer;
                    tam = sizeof (V1alt);
                }
            } else {
                uint64_t * uuid = *(uint64_t**) buffer;
                // TRICK: this code is executed only when V1alt is an IStorage
                // but this is not known at compile time. So compiler complains
                // telling that 'newinstance' does not have the 'setPersistence'
                // method. Thus we call a 'templatized function' to hide this.
                std::get<ix>(values) = instantiateIStorage<V1alt>(sd, uuid);
                // END TRICK.^^^
                tam = sizeof(V1alt);
            }

            setTupleValues<ix+1, restalt...>(sd, (void *)((char*)buffer+tam));
        }

        template <std::size_t ix> void setTupleValues(IStorage* sd, void *buffer) {

        }

        template <class V > char *cast2IStorageBuffer(const V& value,  typename std::enable_if<!std::is_base_of<IStorage, V>::value>::type* =0) {
        }
        template <class V > char *cast2IStorageBuffer(const V& value,  typename std::enable_if<std::is_base_of<IStorage, V>::value>::type* =0) {
            char * buf;
            int size=0;
            size = sizeof(V*);
            buf = (char *) malloc(size);
            IStorage* tmp = static_cast<IStorage*>(const_cast<V*>(&value));// TRICK: We need to static_cast to IStorage to be able to find the base class due to the 'virtual'
            memcpy(buf, &tmp, size);
            return buf;

        }
        template <class V> void createAttributeBuffer(std::string valuetype, const V& value) {
            char * buf;
            int size=0;
            if (ObjSpec::isBasicType(valuetype)) {

                if (typeid(value) == typeid(std::string)){
                    //copy the string so we can pass the address of the copy
                    const std::string& valuestring = reinterpret_cast<const std::string&>(value);
                    const char *valuetmp=valuestring.c_str();
                    char *copyValue=(char *)malloc(strlen(valuetmp) + 1); // TODO: This is NEVER freed.
                    strcpy(copyValue, valuetmp);
                    size = sizeof(copyValue);
                    buf = (char *) malloc(size);
                    memcpy(buf, &copyValue, size);
                } else {
                    size=sizeof(value);
                    buf = (char *) malloc(size);
                    memcpy(buf, &value, size);
                }

            } else { // IStorage*
                //WARNING: this else should compile for any possible class V !!
                // We need to copy the address of the reference.
                // TRICK: this code is executed only when V1alt is an IStorage
                // but this is not known at compile time. So compiler complains
                // about doing wrong castings (casting an IStorage* to an int*).
                // Thus we call a 'templatized function' to hide this.
                buf = cast2IStorageBuffer<V>(value);
                // END TRICK ^^^
                size= sizeof(V*);
            }
            std::pair <char *, int> bufinfo(buf,size);
            valuesTmpBuffer.push_back(bufinfo);
            total_size += size;
        }

        void createValuesBuffer() {
            valuesBuffer = (char *) malloc(total_size);
            char *pBuffer = valuesBuffer;
            for (std::vector <std::pair<char *, int>>::iterator it=valuesTmpBuffer.begin(); it!=valuesTmpBuffer.end(); it++) {
                memcpy(pBuffer, it->first, it->second);
                // Free temporal buffers...
                free(it->first);
                pBuffer+=it->second;
            }
        } 

        void generateAttrDescr(std::string attrBaseName) {
            if (managedValues == 0) {
                std::vector<const char *> v={typeid(V1).name(),typeid(rest).name()...};
                std::vector<const char *>::iterator it = v.begin();
                while (it != v.end()) {
                    std::pair <std::string,std::string> valuedesc (attrBaseName+std::to_string(managedValues), ObjSpec::c_to_cass(*it));
                    valuesDesc.push_back(valuedesc);
                    managedValues++;
                    ++it;
                }
            }

        }

        std::vector<std::pair<std::string, std::string>> getValuesDesc(std::string attrBaseName) {
            if (managedValues == 0) {
                generateAttrDescr(attrBaseName);
            }
            return valuesDesc;
        }

        bool operator==(const AttributeClass &v) const {
            return values == v.values;
        }

        bool operator!=(const AttributeClass &v) const {
            return values != v.values;
        }

        bool operator<(const AttributeClass &v) const {
            return values < v.values;
        }

        // comparator
        bool operator>(const AttributeClass &v) const {
            return values > v.values;
        }

        // comparator
        bool operator<=(const AttributeClass &v) const {
            return values <= v.values;
        }

        // comparator
        bool operator>=(const AttributeClass &v) const {
            return values >= v.values;
        }

        template <std::size_t ix, class ...Types> static typename std::tuple_element<ix, std::tuple<Types...>>::type& get(AttributeClass<Types...>& v){
            return std::get<ix>(v.values);
        }

        char *getValuesBuffer() {
            return valuesBuffer;
        }
        void setValuesBuffer(char *valuesBuffer) {
            this->valuesBuffer = valuesBuffer;
        }
        void setTotalSize(int size) {
            this->total_size = size;
        }

    protected:
        //valuesDesc is used to generate the python definition of the class and to store the attributes description in Cassandra
        std::vector<std::pair<std::string, std::string>> valuesDesc;
        int32_t managedValues=0;
        std::vector <std::pair<char *, int>> valuesTmpBuffer;
        int total_size = 0;
        IStorage *sd = nullptr;
        char *pendingKeysBuffer=nullptr;
        int pendingKeysBufferSize=0;
        char *valuesBuffer = nullptr;


        std::tuple <V1, rest...> values;


};

#endif
