#ifndef OBJSPEC_H
#define OBJSPEC_H

#include <vector>
#include <map>
#include <string>
#include <unordered_set>
#include <cxxabi.h>

#include "configmap.h"

#include <iostream>

class ObjSpec {

public:
	enum valid_types {
		STORAGEOBJ_TYPE,
		STORAGEDICT_TYPE,
		STORAGENUMPY_TYPE,
	};

    // cassandra_types defines the cassandra types used (useful for using it in a switch)
    enum cassandra_types {
        ctNotDefined,
        ctBoolean,
        ctDouble,
        ctInt,
        ctBigint,
        ctFloat,
        ctText
    };

    static enum cassandra_types string2CassandraType(const std::string& cass_type) ;

	std::string table_attr; // String to use in table creation with the attributes (keys+cols)

    ObjSpec();
    ObjSpec(enum valid_types type, std::vector<std::pair<std::string, std::string>>partK, std::vector<std::pair<std::string, std::string>> clustK, std::vector<std::pair<std::string, std::string>> c,std::string pythonSpecString);

    std::string getKeysStr(void) ;
    std::string getColsStr(void) ;
    std::vector<config_map>* getKeysNamesDict(void);
    std::vector<config_map>* getColsNamesDict(void) ;
    std::string debug();
    enum valid_types getType();

    const std::string& getIDModelFromCol(int pos) const;
    const std::string& getIDModelFromKey(int pos) const;
    const std::string& getIDModelFromColName(const std::string & name) const;
    const std::string& getIDObjFromCol(int pos) const;
    const std::string& getIDObjFromKey(int pos) const;
    const std::string& getPythonString()const;
    bool isStream(void) const;
    void enableStream(void) ;
    void disableStream(void) ;
    static bool isBasicType(std::string attr_type);
    static std::string yaml_to_cass(const std::string attr_type);
    static std::string c_to_cass(const std::string attr_type);
    static std::string cass_to_hecuba(const std::string& attr_type);
private:
    enum valid_types objtype;
    bool stream_enabled = false;
    std::vector<std::pair<std::string, std::string>> partitionKeys;
    std::vector<std::pair<std::string, std::string>> clusteringKeys;
    std::vector<std::pair<std::string, std::string>> cols;
    std::string pythonSpecString;

    static std::unordered_set<std::string> basic_types_str;
    static std::unordered_set<std::string> valid_types_str;
    static std::map<std::string, std::string> yaml_to_cass_conversion;
    static std::map<std::string, std::string> c_to_cass_conversion;
    static std::map<std::string, std::string> cass_to_hecuba_conversion;
    // Map to associate the strings with the enum values
    static std::map<std::string, ObjSpec::cassandra_types> mapString2CassandraType;
    //static std::unordered_set<std::string> basic_types_str = {
    //                "counter",
    //                "text",
    //                "boolean",
    //                "decimal",
    //                "double",
    //                "int",
    //                "bigint",
    //                "blob",
    //                "float",
    //                "timestamp",
    //                "time",
    //                "date",
    //                //TODO "list",
    //                //TODO "set",
    //                //TODO "map",
    //                //TODO "tuple"
    //};

    //static std::unordered_set<std::string> valid_types_str = {
    //                //TODO "dict",
    //                "hecuba.hnumpy.StorageNumpy"     //numpy.ndarray
    //};

    void generateTableAttr();
    std::string getCassandraType(std::string type);
};
#endif /* OBJSPEC_H */
