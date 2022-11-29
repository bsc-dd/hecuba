#ifndef _STORAGEDICT_
#define _STORAGEDICT_

#include <map>
#include <iostream>
#include <type_traits>
#include <hecuba/ObjSpec.h>
#include <hecuba/debug.h>
#include <hecuba/IStorage.h>
#include <hecuba/KeyClass.h>
#include <hecuba/ValueClass.h>
#include "UUID.h"



template<class K, class V>

class StorageDict:public IStorage {
public:
    void initObjSpec() {
	K key;
	V v;
	partitionKeys=key.getPartitionKeys();
	clusteringKeys=key.getClusteringKeys();
	values=v.getValues();
	ObjSpec dictSpec;
	dictSpec=ObjSpec(ObjSpec::valid_types::STORAGEDICT_TYPE, partitionKeys, clusteringKeys, values,"");
 	setObjSpec(dictSpec);
    }

    StorageDict() {
	initObjSpec();
    }
    // c++ only calls implicitly the constructor without parameters. To invoke this constructor we need to add to the user class an explicit call to this
    StorageDict(const std::string name) {
	initObjSpec();
	id_obj=name;
	pending_to_persist=true;
    }
    
    ~StorageDict() {}

    V &operator[](K &key) {
	//V v(this, key.getKeysBuffer());
	V *v = new V(this, key.getKeysBuffer());
        return *v;
    }


    // It generates the python specification for the class during the registration of the object
    void generatePythonSpec() {
	std::string pythonSpec = "from hecuba import StorageDict\n\nclass " 
				  + getClassName() + "(StorageDict):\n"
				  + "   '''\n   @TypeSpec dict <<";

	std::string itemSpec = "";
	for (std::vector<std::pair<std::string,std::string>>::iterator it=partitionKeys.begin(); it!=partitionKeys.end(); ++it)
		itemSpec+=it->first + ":"+ it->second + ",";
	for (std::vector<std::pair<std::string,std::string>>::iterator it=clusteringKeys.begin(); it!=clusteringKeys.end(); ++it)
		itemSpec+=it->first + ":"+ it->second + ",";

	pythonSpec += itemSpec.substr(0, itemSpec.size()-1) + ">,"; // replace the last , with a >

	itemSpec = "";
	for (std::vector<std::pair<std::string,std::string>>::iterator it=values.begin(); it!=values.end(); ++it)
		itemSpec+=it->first + ":"+ it->second + ",";

	pythonSpec += itemSpec.substr(0, itemSpec.size()-1) + ">\n   '''\n"; // replace the last , with a >

	setPythonSpec(pythonSpec);
    }
    void assignTableName(std::string id_obj, std::string id_model) {
	this->setTableName(id_obj); //in the case of StorageObject this will be the name of the class
    }
    void persist_metadata(uint64_t* c_uuid) {
	ObjSpec oType = getObjSpec(); 
    	std::string insquery = 	std::string("INSERT INTO ") +
                        	std::string("hecuba.istorage") +
                        	std::string("(storage_id, name, class_name, primary_keys, columns)") +
                        	std::string("VALUES ") +
                        	std::string("(") +
                        	UUID::UUID2str(c_uuid) + std::string(", ") +
                        	"'" + getCurrentSession()->config["execution_name"] + "." + getTableName() + "'" + std::string(", ") +
                        	"'" + this->getClassName() + "'" + std::string(", ") +
                        	oType.getKeysStr() + std::string(", ") +
                        	oType.getColsStr() +
                        	std::string(")");
        getCurrentSession()->run_query(insquery);
    }

private:
    //Istorage * sd;
    std::map<K,V> sd;
    std::vector<std::pair<std::string, std::string>> partitionKeys;
    std::vector<std::pair<std::string, std::string>> clusteringKeys;
    std::vector<std::pair<std::string, std::string>> values;
    
};

#endif
