#ifndef _ATTR_CLASS_
#define _ATTR_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>
#include <tuple>

#include <hecuba/debug.h>
#include "IStorage.h"



template <class V1, class...rest>
class AttributeClass {

public:
    AttributeClass() =default; 

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

    template <std::size_t ix, class V1alt, class...restalt> void setTupleValues(void *buffer) {

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
		std::get<ix>(values)=*(V1alt *)buffer;
		tam = sizeof(V1alt);
	}
	
	setTupleValues<ix+1, restalt...>((void *)((char*)buffer+tam));
    }

    template <std::size_t ix> void setTupleValues(void *buffer) {

    }


       template <class V> void createAttributeBuffer(std::string valuetype, const V& value) {
            char * buf;
            int size=0;
            if (ObjSpec::isBasicType(valuetype)) {

		if (typeid(value) == typeid(std::string)){
                        //copy the string so we can pass the address of the copy
			const std::string& valuestring = reinterpret_cast<const std::string&>(value);
                        const char *valuetmp=valuestring.c_str();
                        char *copyValue=(char *)malloc(strlen(valuetmp) + 1);
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

template <std::size_t ix, class ...Types> static typename std::tuple_element<ix, std::tuple<Types...>>::type& get(AttributeClass<Types...>&v) {
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
