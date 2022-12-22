#ifndef _KEY_CLASS_
#define _KEY_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>
#include <hecuba/debug.h>
#include <hecuba/ObjSpec.h>
#include <hecuba/ObjSpec.h>
#include "AttributeClass.h"


template <class K1, class...rest>

class KeyClass:public AttributeClass<K1,rest...> {
    //First _KeyClass;
    //KeyClass<Rest...> _nextKeyClass;
public:
    KeyClass() = default;

    // Constructor without specifying names.
    KeyClass(K1 part, rest... clustering):AttributeClass<K1,rest...>("keyname",part,clustering...) {
	setKeys(); // the first one of the attributes is the partition key, and the rest the clustering keys
    }

    KeyClass(const KeyClass &w):AttributeClass<K1,rest...>(){
	this->partitionKeys=w.partitionKeys;
	this->clusteringKeys=w.clusteringKeys;
	this->total_size=w.total_size;
	this->valuesBuffer = (char *) malloc (this->total_size);
	memcpy(this->valuesBuffer, w.valuesBuffer,this->total_size);
    }

    // copy assignment
    void operator=(const KeyClass &w) {
	this->partitionKeys=w.partitionKeys;
	this->clusteringKeys=w.clusteringKeys;
	this->total_size=w.total_size;
	this->valuesBuffer = (char *) malloc (this->total_size);
	memcpy(this->valuesBuffer, w.valuesBuffer,this->total_size);
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

    bool operator==(const KeyClass &k) const {
        return true;
    }
    bool operator<(const KeyClass &k) const {
        return false;
    }

    // comparator
    bool operator>(const KeyClass &value) const {
        return false;
    }

    // comparator
    bool operator<=(const KeyClass &value) const {
        return true;
    }

    // comparator
    bool operator>=(const KeyClass &value) const {
        return true;
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
