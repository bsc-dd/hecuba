#include "DataModel.h"

#include <iostream>
#include <algorithm>

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

const std::string& DataModel::getModuleName() const {
    return moduleName;
}

void DataModel::setModuleName(std::string& path) {
    /* path: Path to use.
        Ex: "/home/jcosta/python/app" -> "home.jcosta.python.app"
            "python/app" -> "python.app"
            "./python/app" ->  NOT SUPPORTED '.'
            "../python/app" -> NOT SUPPORTED '.'
     */

    size_t pos = path.find_last_of('.');
    if (pos != std::string::npos) {
		throw std::runtime_error("Tring to set a module name containing dots "+path);
    }
    moduleName = path;
    std::replace( moduleName.begin(), moduleName.end(), '/', '.'); // replace all '/' to '.'
    if (moduleName[0] == '.') { moduleName = moduleName.substr(1, moduleName.size()); }
}
