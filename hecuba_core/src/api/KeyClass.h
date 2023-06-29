#ifndef _KEY_CLASS_
#define _KEY_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>
#include "debug.h"
#include "ObjSpec.h"
#include "AttributeClass.h"
#include "IStorage.h"
#include "HecubaExtrae.h"


template <class K1, class...rest>

class KeyClass:public AttributeClass<K1,rest...> {
    //First _KeyClass;
    //KeyClass<Rest...> _nextKeyClass;
    public:
        KeyClass() = default;

        // Constructor without specifying names.
        KeyClass(const K1& part, const rest&... clustering):AttributeClass<K1,rest...>("keyname",part,clustering...) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SD_KEY|HECUBA_INSTANTIATION);
            setKeys(); // the first one of the attributes is the partition key, and the rest the clustering keys
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        KeyClass(const KeyClass &w):AttributeClass<K1,rest...>(w){
            HecubaExtrae_event(HECUBAEV, HECUBA_SD_KEY|HECUBA_INSTANTIATION);
            this->partitionKeys=w.partitionKeys;
            this->clusteringKeys=w.clusteringKeys;
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }


        // Constructor used by the key iterator in StorageDict: given the buffer with the content of the key we construct a keyclass
        KeyClass(IStorage *sd, char *keysBuffer) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SD_KEY|HECUBA_INSTANTIATION);
            std::pair<short unsigned int, short unsigned int> keys_size = sd->getDataWriter()->get_metadata()->get_keys_size();
            this->total_size=keys_size.first + keys_size.second;
            this->partitionKeys=sd->getPartitionKeys();
            this->clusteringKeys=sd->getClusteringKeys();
            this->managedValues = this->partitionKeys.size() + this->clusteringKeys.size();
            this->valuesDesc=this->partitionKeys;
            this->valuesDesc.insert(this->valuesDesc.end(), this->clusteringKeys.begin(), this->clusteringKeys.end());
            this->valuesBuffer = keysBuffer;
            this->template setTupleValues<0,K1,rest...>(sd, this->valuesBuffer);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        // copy assignment
        void operator=(const KeyClass &w) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SD_KEY|HECUBA_ASSIGNMENT);
            this->partitionKeys=w.partitionKeys;
            this->clusteringKeys=w.clusteringKeys;
            this->valuesDesc=w.valuesDesc;
            this->total_size=w.total_size;
            this->managedValues=w.managedValues;
            this->valuesBuffer = (char *) malloc (this->total_size);
            memcpy(this->valuesBuffer, w.valuesBuffer,this->total_size);
            this->values = w.values;
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        K1 getPartitionKey() {
            //DEBUG("Partition key "<< partitionKey << std::endl);
            return this->partitionKey;             
        }

        std::vector<std::pair<std::string, std::string>> getPartitionKeys() {
            if (this->managedValues == 0) {
                this->generateAttrDescr("keyname");
                setKeys();
            }
            return this->partitionKeys;
        }
        std::vector<std::pair<std::string, std::string>> getClusteringKeys() {
            if (this->managedValues == 0) {
                this->generateAttrDescr("keyname");
                setKeys();
            }
            return this->clusteringKeys;
        }

        void setKeys() {
            std::vector<std::pair<std::string,std::string >>::iterator it = this->valuesDesc.begin();
            while (it != this->valuesDesc.end()) {
                if (it == this->valuesDesc.begin()) {
                    this->partitionKeys.push_back(*it);
                } else {
                    this->clusteringKeys.push_back(*it);
                }
                this->managedValues++;
                ++it;
            }
        }


        char *getKeysBuffer() {
            return this->getValuesBuffer();
        }
        int getTotalSize() {
            return this->total_size;
        }

        std::vector<std::pair<std::string, std::string>> partitionKeys;
        std::vector<std::pair<std::string, std::string>> clusteringKeys;


};

#endif
