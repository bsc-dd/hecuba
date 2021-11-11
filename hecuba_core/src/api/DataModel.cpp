#include "DataModel.h"

#include <iostream>

DataModel::DataModel() {
}

DataModel::~DataModel() {
}

void DataModel::addObjSpec(enum valid_types objtype, std::string id, std::vector<std::pair<std::string,std::string>> keystypes, std::vector<std::pair<std::string,std::string>> colstypes) {
	std::vector<std::pair<std::string,std::string>> empty;
	this->addObjSpec(objtype, id, keystypes, empty, colstypes);
}

std::string DataModel::getCassandraType(std::string type) {
    /** Transform a user 'type' into a Cassandra equivalent type. Basically to store uuids if not a basic type. For example: 'numpy.ndarray' --> 'uuid' */
    std::string res;
    if (basic_types_str.count(type)>0) { //if (basic_types_str.contains(type))
        res = type;
    } else if (valid_types_str.count(type)>0) { //if (valid_types_str.contains(type))
        res = "uuid";
    } else {
        res = "NOT SUPPORTED TYPE [" + type + "]"; // TODO Will be 'uuid'
    }
    return res;
}

void DataModel::addObjSpec(enum valid_types objtype, std::string id, std::vector<std::pair<std::string,std::string>> partitionkeystypes, std::vector<std::pair<std::string,std::string>> clusteringkeystypes, std::vector<std::pair<std::string,std::string>> colstypes) {
	obj_spec o;
	o.objtype = objtype;
	o.partitionKeys = partitionkeystypes;
	o.clusteringKeys = clusteringkeystypes;
	o.cols = colstypes;

	// Generate attributes string for cassandra table creation
	// Ex: "( key1 type1,  key2 type2, key3 type3, col1 type4 ) PRIMARY KEY ( (key1, key2), key3 )"
	std::string attributes ="";
	std::string pkey_str ="";
	std::string ckey_str ="";
	std::string cols_str ="";
    std::vector<std::pair<std::string, std::string>>::iterator it;
	for( it = o.partitionKeys.begin(); it != o.partitionKeys.end(); ) {
		pkey_str += it->first + " " + getCassandraType(it->second);
        it ++;
        if (it != o.partitionKeys.end()) {
		    pkey_str += ", ";
        }
	}
    if (o.clusteringKeys.size() > 0) pkey_str += ", ";

	for( it = o.clusteringKeys.begin(); it != o.clusteringKeys.end(); ) {
		ckey_str += it->first + " " + getCassandraType(it->second);
        it ++;
        if (it != o.clusteringKeys.end()) {
		    ckey_str += ", ";
        }
	}
    if (o.cols.size() > 0) ckey_str += ", ";

	for( it = o.cols.begin(); it != o.cols.end(); ) {
		cols_str += it->first + " " + getCassandraType(it->second);
        it ++;
        if (it != o.cols.end()) {
		    cols_str += ", ";
        }
	}
	attributes = pkey_str + ckey_str + cols_str;

	o.table_attr = " (" + attributes + ", PRIMARY KEY ( ";

    pkey_str = "";

	if (o.partitionKeys.size() > 1) {
		pkey_str = "(";
	}
	for( it = o.partitionKeys.begin(); it != o.partitionKeys.end(); ) {
		pkey_str += it->first;
        it++;
        if (it != o.partitionKeys.end()) {
		    pkey_str += ", ";
        }
	}
	if (o.partitionKeys.size() > 1) {
		pkey_str += ")";
	}

    if (o.clusteringKeys.size() >0) pkey_str += ", ";

	for( it = o.clusteringKeys.begin(); it != o.clusteringKeys.end(); ) {
		pkey_str += it->first;
        it++;
        if (it != o.clusteringKeys.end()) {
		    pkey_str += ", ";
        }
	}
	pkey_str += ")"; // END PRIMARY KEY
	pkey_str += ")"; // END KEYS


	o.table_attr += pkey_str;


	dataModel.insert({id, o});
}

DataModel::obj_spec& DataModel::getObjSpec(std::string id) {
    auto search = dataModel.find(id);
    if (search == dataModel.end()) {
		throw std::runtime_error("Object type "+id+" NOT found. Exiting");
    }
    //std::cout << "Found " << search->first << " " << search->second << '\n';
	return search->second; //return dataModel[id]
}

std::string DataModel::debug() const {
    std::string res = "";

    for (auto it : dataModel) {
        std::cout<<"it.first: " << it.first << std::endl;
        res += "name: " + it.first + " -> " + it.second.debug();
    }

    return res;
}
