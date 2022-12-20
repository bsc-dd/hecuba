#ifndef _VALUE_CLASS_
#define _VALUE_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>
#include <tuple>

#include <hecuba/debug.h>
#include "IStorage.h"



template <class V1, class...rest>

class ValueClass {
    //First _KeyClass;
    //KeyClass<Rest...> _nextKeyClass;
public:
    // Constructor called when instantiating a new value without parameters: MyValueClass v;
    ValueClass() =default; 

    // Constructor called when instantiating a new value with parameters: MyValueClass v(value);
   ValueClass(V1 part, rest... vals) {
        //DEBUG("Number of clustering keys: "<<sizeof...(values)<<std::endl);
        // first element is the partition key, the rest of elements are clustering keys
        // only basic classes are suported
        manageValue<V1>(part);
        manageRest(vals...);
	createValuesBuffer();
	values=std::make_tuple(part, vals...);
   }

    // Constructor called when assigning another value during the instatiation: MyValueClass v = otherValue or MyValueClass v = d[k]
    ValueClass(const ValueClass &w){
        valuesDesc=w.valuesDesc;
	sd = w.sd;
	
	if (w.pendingKeysBuffer != nullptr) {
		// case myvalueclass v =d[k]
		std::cout << "GetItem: move assignemnt to reference to object" << std::endl;
		//valuesBuffer= (char *)malloc(total_size);
		total_size=sd->getDataWriter()->get_metadata()->get_values_size();
		managedValues = w.managedValues;
		if (managedValues == 1) {
			valuesBuffer= (char *)malloc(total_size);
			sd->getItem(w.pendingKeysBuffer,valuesBuffer);
		} else {
			// if multivalue, getitem allocates the space
			sd->getItem(w.pendingKeysBuffer,&valuesBuffer);
		}
		free(w.pendingKeysBuffer);
		pendingKeysBuffer = nullptr;
		pendingKeysBufferSize=0;
		setTupleValues<0,V1,rest...>(valuesBuffer);
	} else {
		// case myvalueclass v = j copy constructor
    		managedValues = w.managedValues;
		total_size = w.total_size;
		valuesBuffer = (char *) malloc(total_size);
		memcpy(valuesBuffer, w.valuesBuffer, total_size);
		values = w.values;
	}

    }

   //Constructor called by the operator [] of StorageDict
    ValueClass(IStorage *sd,char *keysBuffer, int bufferSize) {
	this->sd = sd;
	this->pendingKeysBuffer = (char *) malloc(bufferSize);
	this->pendingKeysBufferSize = bufferSize;
	this->valuesDesc=this->sd->getValuesDesc();
	this->total_size=sd->getDataWriter()->get_metadata()->get_values_size();
	this->managedValues=this->valuesDesc.size();
	memcpy(this->pendingKeysBuffer, keysBuffer, bufferSize);
    }

    ValueClass &operator = (ValueClass & w) {
	//  case v=sd[k]
	if (w.pendingKeysBuffer != nullptr) {
		sd=w.sd;
		total_size=w.total_size;
		std::cout << "GetItem: move assignemnt to reference to object" << std::endl;
		managedValues = w.managedValues;
		if (managedValues == 1) {
			valuesBuffer= (char *)malloc(total_size);
			sd->getItem(w.pendingKeysBuffer,valuesBuffer);
		} else {
			// if multivalue, getitem allocates the space
			sd->getItem(w.pendingKeysBuffer,&valuesBuffer);
		}
		free(w.pendingKeysBuffer);
		pendingKeysBuffer = nullptr;
		w.pendingKeysBuffer=nullptr;
		pendingKeysBufferSize=0;
		w.pendingKeysBufferSize=0;
        	valuesDesc=w.valuesDesc;
		setTupleValues<0,V1,rest...>(valuesBuffer);
		// TODO set values interpreting valuesBuffer
		w.sd=nullptr;
		w.managedValues=0;
		w.total_size=0;
	} else {
		//  case sd[k]=v;
		std::cout << "SetItem: copy assignemnt to reference to object" << std::endl;
		if (pendingKeysBuffer != nullptr) {
			sd->setItem(pendingKeysBuffer,w.valuesBuffer);
			free(pendingKeysBuffer);
			pendingKeysBuffer = nullptr;
			pendingKeysBufferSize=0;
        		valuesDesc=w.valuesDesc;
    			managedValues = w.managedValues;
			valuesBuffer = w.valuesBuffer;
			total_size = w.total_size;
			values = w.values;
		} 
	}

	return *this;

    }

    template <class V> void manageValue(V value) {
            //DEBUG("Key " << key << std::endl);
	    std::string valuetype=ObjSpec::c_to_cass(typeid(decltype(valuetype)).name());
	    std::pair<std::string, std::string> valuedesc("valuename"+std::to_string(managedValues), valuetype);
	    valuesDesc.push_back(valuedesc);
	    createAttributeBuffer <V>(valuetype,value);
    	    managedValues++;
    }

    template <class V1alt, class...restalt> void manageRest(V1alt part, restalt... restValues){
        /* deal with the first parameter */
        //DEBUG("Clust Key " << part << " "<<std::string(typeid(decltype(part)).name())<<std::endl);
	std::string valuetype=ObjSpec::c_to_cass(typeid(decltype(part)).name());
	std::pair<std::string, std::string> valuedesc("valuename"+std::to_string(managedValues), valuetype);
	valuesDesc.push_back(valuedesc);
	createAttributeBuffer <V1alt>(valuetype,part);
    	managedValues++;
        manageRest(restValues...);
    }

    void manageRest() {
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
                        valuesDesc.push_back(valuedesc);
                        managedValues++;
                        ++it;
                }
	}

    }

    std::vector<std::pair<std::string, std::string>> getValuesDesc() {
	if (managedValues == 0) {
		generateValueDescr();
	}
	return valuesDesc;
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

    std::vector<std::pair<std::string, std::string>> valuesDesc;
    int32_t managedValues=0;
    std::vector <std::pair<char *, int>> valuesTmpBuffer;
    int total_size = 0;
    IStorage *sd = nullptr;
    char *pendingKeysBuffer=nullptr;
    int pendingKeysBufferSize=0;
    char *valuesBuffer = nullptr;

    char *getValuesBuffer() {
	return valuesBuffer;
    }
    void setValuesBuffer(char *valuesBuffer) {
	this->valuesBuffer = valuesBuffer;
    }
    void setTotalSize(int size) {
	this->total_size = size;
    }

std::tuple <V1, rest...> values;

template <std::size_t ix, class ...Types> static typename std::tuple_element<ix, std::tuple<Types...>>::type& get(ValueClass<Types...>&v) {
	return std::get<ix>(v.values);
}

};

#endif
