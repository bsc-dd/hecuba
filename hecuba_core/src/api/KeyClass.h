#ifndef _KEY_CLASS_
#define _KEY_CLASS_

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <vector>
#include <hecuba/debug.h>
#include <hecuba/ObjSpec.h>


template <class K1, class...rest>

class KeyClass {
    //First _KeyClass;
    //KeyClass<Rest...> _nextKeyClass;
public:
    KeyClass() = default;

    // Constructor without specifying names.
    KeyClass(K1 part, rest... clustering) {
        //DEBUG("Number of clustering keys: "<<sizeof...(rest)<<std::endl);
        // first element is the partition key, the rest of elements are clustering keys
        // only basic classes are suported
        manageKey<K1>(part);
        manageClustering(clustering...);
	createKeysBuffer();
    }

    KeyClass(const KeyClass &w){
        partitionKey=w.partitionKey;
	partitionKeys=w.partitionKeys;
	clusteringKeys=w.clusteringKeys;
    }

    void operator=(const KeyClass &w) {
        partitionKey=w.partitionKey;
	partitionKeys=w.partitionKeys;
	clusteringKeys=w.clusteringKeys;
    }

    template <class K> void manageKey(K key) {
            //DEBUG("Key " << key << std::endl);
	    std::string keytype=ObjSpec::c_to_cass(typeid(decltype(key)).name());
	    std::pair<std::string, std::string> keydesc("keyname"+std::to_string(managedKeys), keytype);
	    partitionKeys.push_back(keydesc);

	    createAttributeBuffer<K>(keytype, key);
    	    managedKeys++;
            partitionKey = key;
    }

    

    template <class K1alt, class...restalt> void manageClustering(K1alt part, restalt... clustering){
        /* deal with the first parameter */
        //DEBUG("Clust Key " << part << " "<<std::string(typeid(decltype(part)).name())<<std::endl);
	// TODO convert c++ type to cass type

	std::string keytype=ObjSpec::c_to_cass(typeid(decltype(part)).name());
	std::pair<std::string, std::string> keydesc("keyname"+std::to_string(managedKeys), keytype);
	clusteringKeys.push_back(keydesc);
	createAttributeBuffer<K1alt>(keytype, part);
    	managedKeys++;
        manageClustering(clustering...);
    }

    void manageClustering() {
    }


    template <class K> void createAttributeBuffer(std::string keytype, K Key) {
	    char * buf;
            int size=0;
            if (ObjSpec::isBasicType(keytype)) {
		if (typeid(K) == typeid(std::string)) {
			//copy the string so we can pass the address of the copy
			std::string keystring = reinterpret_cast<std::string&>(Key);
			char *keytmp=(char *)keystring.c_str();
			char *copyKey= (char *) malloc(strlen(keytmp) + 1);
			strcpy(copyKey, keytmp);
			size=sizeof(copyKey);
			buf = (char *) malloc(size);
			memcpy(buf, &copyKey, size);
		} else {
                	size=sizeof(Key);
                	buf = (char *) malloc(size);
                	memcpy(buf, &Key, size);
		}
            } else {
                char * Kpointer = (char *) &Key;
                size=sizeof(Kpointer);
                buf = (char *) malloc(size);
                memcpy(buf, &Kpointer, size);
            }
            std::pair <char *, int> bufinfo(buf,size);
            keysTmpBuffer.push_back(bufinfo);
            total_size += size;
    }

    void createKeysBuffer() {
	keysBuffer = (char *) malloc(total_size);
	char *pBuffer = keysBuffer;
	for (std::vector <std::pair<char *, int>>::iterator it=keysTmpBuffer.begin(); it!=keysTmpBuffer.end(); it++) {
		memcpy(pBuffer, it->first, it->second);
		pBuffer+=it->second;
	}
    }


    K1 getPartitionKey() {
        //DEBUG("Partition key "<< partitionKey << std::endl);
        return partitionKey;             
    }
    void  generateKeyDescr() {
	if (managedKeys==0) {
		std::vector<const char *> v={typeid(K1).name(),typeid(rest).name()...};
		std::pair<std::string, std::string> keydesc("keyname"+std::to_string(managedKeys), ObjSpec::c_to_cass(v[0]));
    		managedKeys++;
		partitionKeys.push_back(keydesc);
		std::vector<const char *>::iterator it = v.begin();
    		it++;
		while (it != v.end()) {
			std::pair <std::string,std::string> keydesc ("keyname"+std::to_string(managedKeys), ObjSpec::c_to_cass(*it));
			clusteringKeys.push_back(keydesc);
			managedKeys++;
			++it;
		}
	}
    }
    std::vector<std::pair<std::string, std::string>> getPartitionKeys() {
	if (managedKeys == 0) {
		generateKeyDescr();
	}
	return partitionKeys;
    }
    std::vector<std::pair<std::string, std::string>> getClusteringKeys() {
	if (managedKeys == 0) {
		generateKeyDescr();
	}
	return clusteringKeys;

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
	return keysBuffer;
    }
    int getTotalSize() {
	return total_size;
    }

    K1 partitionKey; 
    std::vector<std::pair<std::string, std::string>> partitionKeys;
    std::vector<std::pair<std::string, std::string>> clusteringKeys;

    std::vector <std::pair<char *, int>> keysTmpBuffer;
    int total_size = 0;

    char *keysBuffer;


    int32_t managedKeys=0;


};

#endif
