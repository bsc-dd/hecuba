#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <vector>
#include <map>
#include <string>
#include <unordered_set>

#include "ObjSpec.h"
#include "configmap.h"

#include <iostream>

class DataModel {
    /* DataModel keeps track of a Data Model, and each data model is formed of
     * a type (Dict, object, numpy) that may contain other data models.
     * For each Data Model we keep: the type name of each field/attribute and its
     * type description (objspec). As we know that this data will be kept in a
     * database we store this information separated in the key/columns format
     * from cassandra.
     * The format used is:
     *      "typename","objspec"
     * where 'nameX' represents a field/attribute type to access inner type.
     * Example:
     *      class dataModel:
     *         '''
     *           dict<lat:double, ts:int>, metrics:numpy.ndarray
     *         '''
     *
     * DataModel:
     *      dataModel objspec1
     *      hecuba.hnumpy.StorageNumpy objspec2
     * And:
     *    objspec1 is:
     *      partitionKeys:  "lat", "double"
     *      clusteringKeys: "ts", "int"
     *      cols:           "metrics", "hecuba.hnumpy.StorageNumpy"
     *    objspec2 is:
     *      partitionKeys:  "storage_id", "uuid",
     *                      "cluster_id", "int"
     *      clusteringKeys: "block_id", "int"
     *      cols:           "payload", "blob"
     */

public:

    struct datamodel_spec {
        ObjSpec o;
        std::string id;
        datamodel_spec(){};
    };

    DataModel();
    ~DataModel();

    void addObjSpec(ObjSpec::valid_types objtype, std::string id, std::vector<std::pair<std::string,std::string>> keystypes, std::vector<std::pair<std::string,std::string>> colstypes);
    void addObjSpec(ObjSpec::valid_types objtype, std::string id, std::vector<std::pair<std::string,std::string>> partitionkeystypes, std::vector<std::pair<std::string,std::string>> clusteringkeystypes, std::vector<std::pair<std::string,std::string>> colstypes);

    void addObjSpec(std::string id, const ObjSpec& o);

    ObjSpec& getObjSpec(std::string id);
    std::string debug() const;
    const std::string& getModuleName() const;
    void setModuleName(std::string& name);
private:
    std::map <std::string, ObjSpec> dataModel;
    std::string moduleName;


};
#endif /* DATAMODEL_H */
