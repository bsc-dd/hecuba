#include <ObjSpec.h>
#include "ModuleException.h"


//Initialize static attributes
// mapString2CassandraType : map to translate from a cassandra string type to a constant value
std::map<std::string, ObjSpec::cassandra_types>ObjSpec::mapString2CassandraType = {
                    { "boolean", ObjSpec::ctBoolean},
                    { "double", ObjSpec::ctDouble},
                    { "int", ObjSpec::ctInt},
                    { "bigint", ObjSpec::ctBigint},
                    { "float", ObjSpec::ctFloat},
                    { "text", ObjSpec::ctText}
    };
std::unordered_set<std::string> ObjSpec::basic_types_str = {
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

std::unordered_set<std::string> ObjSpec::valid_types_str = {
                    //TODO "dict",
                    "hecuba.hnumpy.StorageNumpy"     //numpy.ndarray
};

std::map<std::string, std::string> ObjSpec::yaml_to_cass_conversion {
                    //{ "counter",  "counter"},
                    { "str",      "text"},
                    { "bool",     "boolean"},
                    //{ "???",      "decimal"},
                    { "double",   "double"},
                    { "int",      "int"},
                    { "long",     "bigint"},
                    { "blob",     "blob"},
                    { "float",    "float"},
                    //{ "timestamp","???"},
                    //{ "time",     "time"},
                    //{ "date",     "date"},
                    //{ //TODO "list",
                    //{ //TODO "set",
                    //{ //TODO "map",
                    //{ //TODO "tuple"
};

std::map<std::string, std::string> ObjSpec::c_to_cass_conversion {
                    //{ "counter",  "counter"},
                    { "b",     "boolean"},
                    //{ "???",      "decimal"},
                    { "d",   "double"},
                    { "i",      "int"},
                    { "l",     "bigint"},
                    //{ "blob",     "blob"},
                    { "f",    "float"},
		    {"basic_string", "text"}
                    //{ "timestamp","???"},
                    //{ "time",     "time"},
                    //{ "date",     "date"},
                    //{ //TODO "list",
                    //{ //TODO "set",
                    //{ //TODO "map",
                    //{ //TODO "tuple"
};
std::map<std::string, std::string> ObjSpec::cass_to_hecuba_conversion {
                    { "boolean","bool"},
                    { "double", "double"},
                    { "int",    "int"},
                    { "float",  "float"},
                    { "text",   "str"}
};



ObjSpec::ObjSpec() {}

ObjSpec::ObjSpec(enum valid_types type, std::vector<std::pair<std::string, std::string>>partK, std::vector<std::pair<std::string, std::string>> clustK, std::vector<std::pair<std::string, std::string>> c, std::string pythonString){
    objtype         =type;
    partitionKeys   =partK;
    clusteringKeys  =clustK;
    cols            =c;
    pythonSpecString=pythonString;
    generateTableAttr();
}

std::string ObjSpec::getPythonString() {
    return pythonSpecString;

}
bool ObjSpec::isBasicType(std::string attr_type) {

    return (basic_types_str.count(attr_type)>0);  //if (basic_types_str.contains(type))
}

std::string ObjSpec::yaml_to_cass(const std::string attr_type) {

    std::string res;
    try{
        res = ObjSpec::yaml_to_cass_conversion.at(attr_type);
    } catch( std::out_of_range &e) {
        res = attr_type;
    }
    return res;
}

std::string ObjSpec::c_to_cass(const std::string attr_type) {

	std::string res,type;
	try{
		res = ObjSpec::c_to_cass_conversion.at(attr_type);
	} catch( std::out_of_range &e) {
		int32_t st;
		type =abi::__cxa_demangle(attr_type.c_str(), NULL, NULL, &st);
		if (type.find("basic_string") != std::string::npos)
			res="text";
		else {
		// To keep compatibility with the python layer of hecuba we register StorageNumpys as hecuba.hnumpy.StorageNumpy and we store any other Hecuba object with the
		// fully qualified name. At this moment each class definition is stored in a separate file with the same name as the class, so the FQname is class.class
			if (type == "StorageNumpy") {
				// If the attribute is a StorageNumpy we store in cassandra the fully qualified name used in the python side of Hecuba
				type = std::string("hecuba.hnumpy.StorageNumpy");
			} else {
				if (type.find_first_of(".") == std::string::npos) {
					type = type + "." + type;
				}
			}
			res = type;
		}
	}
	return res;
}

std::string ObjSpec::cass_to_hecuba(const std::string& attr_type) {
    std::string res;
    try {
        res =  ObjSpec::cass_to_hecuba_conversion.at(attr_type);
    } catch (std::out_of_range e) {
        res = attr_type;
    }
    return res;
}


std::string ObjSpec::getKeysStr(void) {
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

std::string ObjSpec::getColsStr(void) {
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

std::vector<config_map>* ObjSpec::getKeysNamesDict(void) {
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

std::vector<config_map>* ObjSpec::getColsNamesDict() {
    // Generate a dictionary to be used by storageInterface->make_writer //TODO: Remove this garbage
    std::vector<config_map>* res = new std::vector<config_map>;
    for(std::vector<std::pair<std::string,std::string>>::iterator it = cols.begin(); it != cols.end(); it++){
        config_map element;
        element["name"] = it->first;
        res->push_back(element);
    }
    return res;
}

ObjSpec::valid_types ObjSpec::getType() {
    return objtype;
}

std::string ObjSpec::getIDModelFromCol(int i) {
    return cols[i].second;
}
std::string ObjSpec::getIDModelFromKey(int i) {
    /* keys are splitted into partitionKeys and clusteringKeys */
    int partKeySize=partitionKeys.size();

    if (i < partKeySize) {
        return partitionKeys[i].second;
    }
    else {
        return clusteringKeys[i-partKeySize].second;
    }
}

const std::string& ObjSpec::getIDModelFromColName(const std::string & name) {
    for(uint16_t i=0; i<cols.size(); i++) {
        if (cols[i].first == name) {
            return cols[i].second;
        }
    }
    throw ModuleException("ObjSpec::getIDModelFromColName Column name "+name+" does not exist");
    return std::string("OK, OK, I'M NOT RETURNING THIS");
}

std::string ObjSpec::getIDObjFromCol(int i) {
    return cols[i].first;
}

std::string ObjSpec::getIDObjFromKey(int i) {
    /* keys are splitted into partitionKeys and clusteringKeys */
    int partKeySize=partitionKeys.size();

    if (i < partKeySize) {
        return partitionKeys[i].first;
    }
    else {
        return clusteringKeys[i-partKeySize].first;
    }
}

std::string ObjSpec::debug() {
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

std::string ObjSpec::getCassandraType(std::string type) {
    /** Transform a user 'type' into a Cassandra equivalent type. Basically to store uuids if not a basic type. For example: 'numpy.ndarray' --> 'uuid' */
    std::string res;
    if (basic_types_str.count(type)>0) { //if (basic_types_str.contains(type))
        res = type;
    } else {
        res = "uuid";
    }
    return res;
}

bool ObjSpec::isStream(void) const {
    return stream_enabled;
}
void ObjSpec::enableStream(void) {
    stream_enabled = true;
}
void ObjSpec::disableStream(void) {
    stream_enabled = false;
}


void ObjSpec::generateTableAttr() {
    // Generate attributes string for cassandra table creation
    // Ex: "( key1 type1,  key2 type2, key3 type3, col1 type4 ) PRIMARY KEY ( (key1, key2), key3 )"
    std::string attributes ="";
    std::string pkey_str ="";
    std::string ckey_str ="";
    std::string cols_str ="";
    std::vector<std::pair<std::string, std::string>>::iterator it;
    for( it = partitionKeys.begin(); it != partitionKeys.end(); ) {
        pkey_str += it->first + " " + getCassandraType(it->second);
        it ++;
        if (it != partitionKeys.end()) {
            pkey_str += ", ";
        }
    }
    if (clusteringKeys.size() > 0) pkey_str += ", ";

    for( it = clusteringKeys.begin(); it != clusteringKeys.end(); ) {
        ckey_str += it->first + " " + getCassandraType(it->second);
        it ++;
        if (it != clusteringKeys.end()) {
            ckey_str += ", ";
        }
    }
    if (cols.size() > 0) ckey_str += ", ";

    for( it = cols.begin(); it != cols.end(); ) {
        cols_str += it->first + " " + getCassandraType(it->second);
        it ++;
        if (it != cols.end()) {
            cols_str += ", ";
        }
    }
    attributes = pkey_str + ckey_str + cols_str;

    table_attr = " (" + attributes + ", PRIMARY KEY ( ";

    pkey_str = "";

    if (partitionKeys.size() > 1) {
        pkey_str = "(";
    }
    for( it = partitionKeys.begin(); it != partitionKeys.end(); ) {
        pkey_str += it->first;
        it++;
        if (it != partitionKeys.end()) {
            pkey_str += ", ";
        }
    }
    if (partitionKeys.size() > 1) {
        pkey_str += ")";
    }

    if (clusteringKeys.size() >0) pkey_str += ", ";

    for( it = clusteringKeys.begin(); it != clusteringKeys.end(); ) {
        pkey_str += it->first;
        it++;
        if (it != clusteringKeys.end()) {
            pkey_str += ", ";
        }
    }
    pkey_str += ")"; // END PRIMARY KEY
    pkey_str += ")"; // END KEYS


    table_attr += pkey_str;
}

// string2CassandraType : translate a cassandra string type to its constant value
ObjSpec::cassandra_types ObjSpec::string2CassandraType(const std::string& cass_type) {
    return mapString2CassandraType.at(cass_type);
}
