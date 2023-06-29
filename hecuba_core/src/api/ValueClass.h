#ifndef _VALUE_CLASS_
#define _VALUE_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>
#include <tuple>

#include "debug.h"
#include "IStorage.h"
#include "AttributeClass.h"
#include "HecubaExtrae.h"



template <class V1, class...rest>
class ValueClass:public AttributeClass<V1,rest...>{ 
    //First _KeyClass;
    //KeyClass<Rest...> _nextKeyClass;
    public:
        // Constructor called when instantiating a new value without parameters: MyValueClass v;
        ValueClass() =default; 

        // Constructor called when instantiating a new value with parameters: MyValueClass v(value);
        ValueClass(const V1& part, rest... vals):AttributeClass<V1,rest...>("valuename",part,vals...) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SD_VALUE|HECUBA_INSTANTIATION);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        // Constructor called when assigning another value during the instatiation: MyValueClass v = otherValue or MyValueClass v = d[k]
        ValueClass(const ValueClass& w): AttributeClass<V1,rest...>(w){
            HecubaExtrae_event(HECUBAEV, HECUBA_SD_VALUE|HECUBA_INSTANTIATION);
            if (w.pendingKeysBuffer != nullptr) {
                // case myvalueclass v =d[k]
                complete_reading();

            } 
            // case else myvalueclass v = j copy constructor implemented in the AttributeClass copy constructor
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        void complete_reading() {
            if (this->managedValues == 1) {
                this->valuesBuffer= (char *)malloc(this->total_size);
                this->sd->getItem(this->pendingKeysBuffer,this->valuesBuffer);
            } else {
                //for more than one attribute getItem allocates the space
                this->sd->getItem(this->pendingKeysBuffer,&this->valuesBuffer);
            }
            this->template setTupleValues<0,V1,rest...>(this->sd, this->valuesBuffer);

        }

        //Constructor called by the operator [] of StorageDict
        ValueClass(IStorage *sd,char *keysBuffer, int bufferSize):AttributeClass<V1,rest...>() {
            HecubaExtrae_event(HECUBAEV, HECUBA_SD_VALUE|HECUBA_SELECTOR);
            this->sd = sd;
            this->pendingKeysBuffer = (char *) malloc(bufferSize);
            this->pendingKeysBufferSize = bufferSize;
            this->valuesDesc=this->sd->getValuesDesc();
            this->total_size=sd->getDataWriter()->get_metadata()->get_values_size();
            this->managedValues=this->valuesDesc.size();
            this->valuesBuffer=nullptr;
            memcpy(this->pendingKeysBuffer, keysBuffer, bufferSize);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        ValueClass &operator = (ValueClass & w) {
            //  case v=sd[k]
            if (w.pendingKeysBuffer != nullptr) {
                HecubaExtrae_event(HECUBAEV, HECUBA_SD|HECUBA_READ);
                this->sd=w.sd;
                this->total_size=w.total_size;
                this->managedValues = w.managedValues;
                this->valuesDesc=w.valuesDesc;
                this->pendingKeysBuffer = w.pendingKeysBuffer;
                this->pendingKeysBufferSize = w.pendingKeysBufferSize;

                complete_reading();

                free(w.pendingKeysBuffer);
                this->pendingKeysBuffer = nullptr;
                w.pendingKeysBuffer=nullptr;
                this->pendingKeysBufferSize=0;
                w.pendingKeysBufferSize=0;
                // TODO set values interpreting valuesBuffer
                w.sd=nullptr;
                this->sd=nullptr;
                w.managedValues=0;
                w.total_size=0;
                w.valuesBuffer=nullptr;

            } else {
                //  case sd[k]=v;
                HecubaExtrae_event(HECUBAEV, HECUBA_SD|HECUBA_WRITE);
                if (this->pendingKeysBuffer != nullptr) {
                    this->sd->setItem(this->pendingKeysBuffer,w.valuesBuffer);
                    free(this->pendingKeysBuffer);
                    this->pendingKeysBuffer = nullptr;
                    this->pendingKeysBufferSize=0;
                    this->sd=nullptr;
                    this->valuesDesc=w.valuesDesc;
                    this->managedValues = w.managedValues;
                    this->total_size = w.total_size;
                    this->valuesBuffer = (char *) malloc(this->total_size);
                    memcpy(this->valuesBuffer, w.valuesBuffer, this->total_size);
                    this->values = w.values;
                } 
            }
            HecubaExtrae_event(HECUBAEV, HECUBA_END);

            return *this;

        }
        template <std::size_t ix, class ...Types> static typename std::tuple_element<ix, std::tuple<Types...>>::type& get(ValueClass<Types...>&v){
            //if we come from d[k], we have to complete the getitem started at the operator []
            if (v.valuesBuffer == nullptr) {
                HecubaExtrae_event(HECUBAEV, HECUBA_SD|HECUBA_READ);
                v.complete_reading();
                HecubaExtrae_event(HECUBAEV, HECUBA_END);
            }
            return std::get<ix>(v.values);
        }

};

#endif
