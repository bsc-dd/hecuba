#ifndef _VALUE_CLASS_
#define _VALUE_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>

#include <hecuba/debug.h>
#include "IStorage.h"



template <class V1, class...rest>

class ValueClass {
    //First _KeyClass;
    //KeyClass<Rest...> _nextKeyClass;
public:
    ValueClass() =default;

    // Constructor without specifying names.
   ValueClass(V1 part, rest... values) {
        //DEBUG("Number of clustering keys: "<<sizeof...(values)<<std::endl);
        // first element is the partition key, the rest of elements are clustering keys
        // only basic classes are suported
        manageValue<V1>(part);
        manageRest(values...);
	createValuesBuffer();
    }

    ValueClass(const ValueClass &w){
        values=w.values;
    }

    ValueClass(IStorage *sd,char *keysBuffer) {
	this->sd = sd;
	this->pendingKeysBuffer = keysBuffer;
    }

    void operator=(const ValueClass &w) {
        values=w.values;
    	managedValues=w.managedValues;
	total_size=w.total_size;
	valuesBuffer=(char *) malloc(total_size);
	memcpy(valuesBuffer,w.valuesBuffer,total_size);
	
    	//valuesTmpBuffer --> is this only necessary to generate the buffer?
	sd->setItem(pendingKeysBuffer,valuesBuffer);

    }

    template <class V> void manageValue(V value) {
            //DEBUG("Key " << key << std::endl);
	    std::string valuetype=ObjSpec::c_to_cass(typeid(decltype(valuetype)).name());
	    std::pair<std::string, std::string> valuedesc("valuename"+std::to_string(managedValues), valuetype);
	    values.push_back(valuedesc);
	    createAttributeBuffer <V>(valuetype,value);
    	    managedValues++;
    }

    template <class V1alt, class...restalt> void manageRest(V1alt part, restalt... restValues){
        /* deal with the first parameter */
        //DEBUG("Clust Key " << part << " "<<std::string(typeid(decltype(part)).name())<<std::endl);
	std::string valuetype=ObjSpec::c_to_cass(typeid(decltype(part)).name());
	std::pair<std::string, std::string> valuedesc("valuename"+std::to_string(managedValues), valuetype);
	values.push_back(valuedesc);
	createAttributeBuffer <V1alt>(valuetype,part);
    	managedValues++;
        manageRest(restValues...);
    }

    void manageRest() {
    }


       template <class V> void createAttributeBuffer(std::string valuetype, V value) {
            char * buf;
            int size=0;
            if (ObjSpec::isBasicType(valuetype)) {

		if (typeid(value) == typeid(std::string)){
                        //copy the string so we can pass the address of the copy
			std::string valuestring = reinterpret_cast<std::string&>(value);
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

            } else {
                char * Vpointer = (char *) &value;
                size=sizeof(Vpointer);
                buf = (char *) malloc(size);
                memcpy(buf, &Vpointer, size);
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

    void generateValueDescr() {
	if (managedValues == 0) {
                std::vector<const char *> v={typeid(V1).name(),typeid(rest).name()...};
                std::vector<const char *>::iterator it = v.begin();
                while (it != v.end()) {
                        std::pair <std::string,std::string> valuedesc ("valuename"+std::to_string(managedValues), ObjSpec::c_to_cass(*it));
                        values.push_back(valuedesc);
                        managedValues++;
                        ++it;
                }
	}

    }

    std::vector<std::pair<std::string, std::string>> getValues() {
	if (managedValues == 0) {
		generateValueDescr();
	}
	return values;
    }

    bool operator==(const ValueClass &v) const {
        return true;
    }
    bool operator<(const ValueClass &v) const {
        return false;
    }

    // comparator
    bool operator>(const ValueClass &value) const {
        return false;
    }

    // comparator
    bool operator<=(const ValueClass &value) const {
        return true;
    }

    // comparator
    bool operator>=(const ValueClass &value) const {
        return true;
    }

    std::vector<std::pair<std::string, std::string>> values;
    int32_t managedValues=0;
    std::vector <std::pair<char *, int>> valuesTmpBuffer;
    int total_size = 0;
    IStorage *sd;
    char *pendingKeysBuffer;
    char *valuesBuffer;

    char *getValuesBuffer() {
	return valuesBuffer;

    }


};

#endif
