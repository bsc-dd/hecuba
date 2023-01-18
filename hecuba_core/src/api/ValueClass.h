#ifndef _VALUE_CLASS_
#define _VALUE_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>
#include <tuple>

#include <hecuba/debug.h>
#include "IStorage.h"
#include "AttributeClass.h"



template <class V1, class...rest>
class ValueClass:public AttributeClass<V1,rest...>{ 
    //First _KeyClass;
    //KeyClass<Rest...> _nextKeyClass;
public:
    // Constructor called when instantiating a new value without parameters: MyValueClass v;
    ValueClass() =default; 

    // Constructor called when instantiating a new value with parameters: MyValueClass v(value);
   ValueClass(const V1& part, rest... vals):AttributeClass<V1,rest...>("valuename",part,vals...) {
   }

    // Constructor called when assigning another value during the instatiation: MyValueClass v = otherValue or MyValueClass v = d[k]
    ValueClass(const ValueClass &w): AttributeClass<V1,rest...>(){
        this->valuesDesc=w.valuesDesc;
	this->sd = w.sd;
	
	if (w.pendingKeysBuffer != nullptr) {
		// case myvalueclass v =d[k]
		//valuesBuffer= (char *)malloc(total_size);
		this->total_size=this->sd->getDataWriter()->get_metadata()->get_values_size();
		this->managedValues = w.managedValues;
		if (this->managedValues == 1) {
			this->valuesBuffer= (char *)malloc(this->total_size);
			this->sd->getItem(w.pendingKeysBuffer,this->valuesBuffer);
		} else {
			// if multivalue, getitem allocates the space
			this->sd->getItem(w.pendingKeysBuffer,&this->valuesBuffer);
		}
		free(w.pendingKeysBuffer);
		this->pendingKeysBuffer = nullptr;
		this->pendingKeysBufferSize=0;
		this->template setTupleValues<0,V1,rest...>(this->valuesBuffer);
	} else {
		// case myvalueclass v = j copy constructor
    		this->managedValues = w.managedValues;
		this->total_size = w.total_size;
		this->valuesBuffer = (char *) malloc(this->total_size);
		memcpy(this->valuesBuffer, w.valuesBuffer, this->total_size);
		this->values = w.values;
	}

    }

   //Constructor called by the operator [] of StorageDict
    ValueClass(IStorage *sd,char *keysBuffer, int bufferSize):AttributeClass<V1,rest...>() {
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
		this->sd=w.sd;
		this->total_size=w.total_size;
		this->managedValues = w.managedValues;
		if (this->managedValues == 1) {
			this->valuesBuffer= (char *)malloc(this->total_size);
			this->sd->getItem(w.pendingKeysBuffer,this->valuesBuffer);
		} else {
			// if multivalue, getitem allocates the space
			this->sd->getItem(w.pendingKeysBuffer,&this->valuesBuffer);
		}
		free(w.pendingKeysBuffer);
		this->pendingKeysBuffer = nullptr;
		w.pendingKeysBuffer=nullptr;
		this->pendingKeysBufferSize=0;
		w.pendingKeysBufferSize=0;
        	this->valuesDesc=w.valuesDesc;
		this->template setTupleValues<0,V1,rest...>(this->valuesBuffer);
		// TODO set values interpreting valuesBuffer
		w.sd=nullptr;
		w.managedValues=0;
		w.total_size=0;
	} else {
		//  case sd[k]=v;
		if (this->pendingKeysBuffer != nullptr) {
			this->sd->setItem(this->pendingKeysBuffer,w.valuesBuffer);
			free(this->pendingKeysBuffer);
			this->pendingKeysBuffer = nullptr;
			this->pendingKeysBufferSize=0;
        		this->valuesDesc=w.valuesDesc;
    			this->managedValues = w.managedValues;
			this->valuesBuffer = w.valuesBuffer;
			this->total_size = w.total_size;
			this->values = w.values;
		} 
	}

	return *this;

    }

};

#endif
