#include "DataModel.h"

#include <iostream>

DataModel::DataModel() {
}

DataModel::~DataModel() {
}

void DataModel::addObjSpec(std::string id, const ObjSpec& o ) {
	dataModel.insert({id, o});
}

void DataModel::addObjSpec(ObjSpec::valid_types objtype, std::string id, std::vector<std::pair<std::string,std::string>> keystypes, std::vector<std::pair<std::string,std::string>> colstypes) {
	std::vector<std::pair<std::string,std::string>> empty;
	this->addObjSpec(objtype, id, keystypes, empty, colstypes);
}

void DataModel::addObjSpec(ObjSpec::valid_types objtype, std::string id, std::vector<std::pair<std::string,std::string>> partitionkeystypes, std::vector<std::pair<std::string,std::string>> clusteringkeystypes, std::vector<std::pair<std::string,std::string>> colstypes) {
	ObjSpec o = ObjSpec(objtype,partitionkeystypes,clusteringkeystypes,colstypes,std::string());
	dataModel.insert({id, o});

}

ObjSpec& DataModel::getObjSpec(std::string id) {
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
