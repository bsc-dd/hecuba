#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <vector>
#include <map>
#include <string>
#include <unordered_set>

#include "configmap.h"

#include <iostream>

class DataModel {
    /* DataModel keeps track of a Data Model, and each data model is formed of
     * a type (Dict, object, numpy) that may contain other data models.
     * For each Data Model we keep: the names of each field/attribute and its
     * type. As we know that this data will be kept in a database we store this
     * information separated in the key/columns format from cassandra.
     * The format used is:
     *      "name3","type"
     * where 'nameX' represents a field/attribute name to access inner type.
     * Example:
     *      dict<lat:string, ts:int> dict<key0:int,key1:int> metrics:numpy
     *
     *      keystypes:"lat","string"
     *                "ts","int"
     *                ".key0","int"    ## Internal use of 'empty string' to identify unnamed fields (dict only)
     *                ".key1","int"
     *      colstypes:"", "dict"
     *                ".metrics", "numpy"
     */

public:
	enum valid_types {
		STORAGEOBJ_TYPE,
		STORAGEDICT_TYPE,
		STORAGENUMPY_TYPE
	};

    struct obj_spec{
		std::string table_attr;	// String to use in table creation with the attributes (keys+cols)
		enum valid_types objtype;
		std::vector<std::pair<std::string, std::string>> partitionKeys;
		std::vector<std::pair<std::string, std::string>> clusteringKeys;
		std::vector<std::pair<std::string, std::string>> cols;

        std::string getKeysStr(void) {
            // Returns a string with a list of tuples name+type. Example: [('lat','int'), ('ts','int')]
            std::vector<std::pair<std::string,std::string>>::iterator it;
            std::string res = "[";
            for(it = partitionKeys.begin(); it != partitionKeys.end(); ){
                res = res + "('" + it->first + "','" + it->second + "')";
                ++it;
                if (it != partitionKeys.end()) {
                    res += ", ";
                }
            }
            if (!clusteringKeys.empty()) {
                    res += ", ";
            }
            for(it = clusteringKeys.begin(); it != clusteringKeys.end(); ){
                res = res + "('" + it->first + "','" + it->second + "')";
                ++it;
                if (it != clusteringKeys.end()) {
                    res += ", ";
                }
            }
            res += "]";
            return res;
        }
        std::string getColsStr(void) {
            // Returns a string with a list of tuples name+type. Example: [('lat','int'), ('ts','int')]
            std::vector<std::pair<std::string,std::string>>::iterator it;
            std::string res = "[";
            for(it = cols.begin(); it != cols.end(); ){
                res = res + "('" + it->first + "','" + it->second + "')";
                ++it;
                if (it != cols.end()) {
                    res += ", ";
                }
            }
            res += "]";
            return res;
        }

        std::vector<config_map>* getKeysNamesDict(void) {
            // Generate a dictionary to be used by storageInterface->make_writer //TODO: Remove this garbage
            std::vector<config_map>* res = new std::vector<config_map>;
            for(std::vector<std::pair<std::string,std::string>>::iterator it = partitionKeys.begin(); it != partitionKeys.end(); it++){
                config_map element;
                element["name"] = it->first;
                res->push_back(element);
            }
            for(std::vector<std::pair<std::string,std::string>>::iterator it = clusteringKeys.begin(); it != clusteringKeys.end(); it++){
                config_map element;
                element["name"] = it->first;
                res->push_back(element);
            }
            return res;
        }
        std::vector<config_map>* getColsNamesDict() {
            // Generate a dictionary to be used by storageInterface->make_writer //TODO: Remove this garbage
            std::vector<config_map>* res = new std::vector<config_map>;
            for(std::vector<std::pair<std::string,std::string>>::iterator it = cols.begin(); it != cols.end(); it++){
                config_map element;
                element["name"] = it->first;
                res->push_back(element);
            }
            return res;
        }
        std::string debug() {
            std::string res;
            switch(objtype) {
                case STORAGEOBJ_TYPE:
                    res = "STORAGEOBJ";
                    break;
                case STORAGEDICT_TYPE:
                    res = "STORAGEDICT";
                    break;
                case STORAGENUMPY_TYPE:
                    res = "STORAGENUMPY";
                    break;
                default:
                    res = "UNKNOWN";
            }
            res += " " + getKeysStr() + getColsStr();
            return res;
        }
	};

    DataModel();
    ~DataModel();

    void addObjSpec(enum valid_types objtype, std::string id, std::vector<std::pair<std::string,std::string>> keystypes, std::vector<std::pair<std::string,std::string>> colstypes);
    void addObjSpec(enum valid_types objtype, std::string id, std::vector<std::pair<std::string,std::string>> partitionkeystypes, std::vector<std::pair<std::string,std::string>> clusteringkeystypes, std::vector<std::pair<std::string,std::string>> colstypes);

	obj_spec& getObjSpec(std::string id);
    std::string debug() const;

private:
    std::map <std::string, obj_spec> dataModel;


    std::string getCassandraType(std::string);

    std::unordered_set<std::string> basic_types_str = {
                    "counter",
                    "text",
                    "boolean",
                    "decimal",
                    "double",
                    "int",
                    "bigint",
                    "blob",
                    "float",
                    "timestamp",
                    "time",
                    "date",
                    //TODO "list",
                    //TODO "set",
                    //TODO "map",
                    //TODO "tuple"
    };
    std::unordered_set<std::string> valid_types_str = {
                    //TODO "dict",
                    "hecuba.hnumpy.StorageNumpy"     //numpy.ndarray
    };
};
#endif /* DATAMODEL_H */
