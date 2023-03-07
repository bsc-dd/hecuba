#ifndef _ATTR_CLASS_
#define _ATTR_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>
#include <tuple>

#include <hecuba/debug.h>
#include "IStorage.h"
#include "StorageDict.h"



template <class V1, class...rest>
class AttributeClass {

public:
    AttributeClass() =default; 

	AttributeClass(const AttributeClass& a) {
		*this = a;
	}

    ~AttributeClass() {
		if (valuesBuffer != nullptr) {
			free(valuesBuffer);
			valuesBuffer = nullptr;
		}
		if (pendingKeysBuffer != nullptr) {
			free(pendingKeysBuffer);
			pendingKeysBuffer = nullptr;
		}
    };
    // Constructor called when instantiating a new value with parameters: MyValueClass v(value);
   AttributeClass(std::string attrBaseName, const V1& part, rest... vals) {
        //DEBUG("Number of clustering keys: "<<sizeof...(values)<<std::endl);
        // first element is the partition key, the rest of elements are clustering keys
        // only basic classes are suported
        manageAttr<V1>(attrBaseName,part);
        manageRest(attrBaseName,vals...);
        createValuesBuffer();
        values=std::make_tuple(part, vals...);
   }

   AttributeClass& operator = (const AttributeClass& a) {
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

        return *this;
   }

    template <class V> void manageAttr(std::string attrBaseName, const V& value) {
            DBG("Clust Key " << value << " "<<std::string(typeid(decltype(value)).name())<<std::endl);
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
		// method. Thus we cast the newinstance to force to be considered as an IStorage.
		V1alt newinstance;
		sd->getCurrentSession()->registerObject((IStorage*)&newinstance);
		((IStorage*)&newinstance)->setPersistence(valuesDesc[ix].second, uuid); // registerObject + get name from cassandra + create writer and cache
		// END TRICK.^^^
		std::get<ix>(values)=newinstance;
		tam = sizeof(V1alt);
	}
	
	setTupleValues<ix+1, restalt...>(sd, (void *)((char*)buffer+tam));
    }

    template <std::size_t ix> void setTupleValues(IStorage* sd, void *buffer) {

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
                size = sizeof(V*);
                buf = (char *) malloc(size);
		// We need to copy the address of the reference (this make the trick)
		const void *tmp = &value; 
        	memcpy(buf, &tmp, size);
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
