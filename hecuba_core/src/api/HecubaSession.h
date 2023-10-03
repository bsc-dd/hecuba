#ifndef HECUBA_SESSION_H
#define HECUBA_SESSION_H

#include <fstream>
#include <iostream>
#include "StorageInterface.h"
#include "configmap.h"
#include "ArrayDataStore.h"
#include <mutex>

class HecubaSession {
    /** Establish connection with Underlying storage system */
public:
    static HecubaSession& get();
    HecubaSession(HecubaSession const&)              = delete;
    void operator=(HecubaSession const&)             = delete;

    config_map config;
    std::shared_ptr<StorageInterface> getStorageInterface() {
        return storageInterface;
    }; /* Connection to Cassandra */
    std::string getExecutionName();
    CassError run_query(std::string) const;

    Writer * getNumpyMetaWriter() const;
    CacheTable * getHecubaIstorageAccess() const;

    bool registerObject(const std::shared_ptr<CacheTable> c, const std::string& class_name) ;
    bool registerObject(const std::shared_ptr<ArrayDataStore> a, const std::string& class_name) ;
    bool registerClassName(const std::string& class_name);
    int wait_writes_completion(void); /* Wait for the finalization of any pending write operation */
private:

    std::mutex mxalive_objects;
    std::list<std::shared_ptr<CacheTable>> alive_objects; //List of registered objects with pending writes
    std::mutex mxalive_numpy_objects;
    std::list<std::shared_ptr<ArrayDataStore>> alive_numpy_objects; //List of registered numpy objects with pending writes

    std::map<std::string,char> registeredClasses; // Map of classes with at least one intance occurrence: to detect if it is necessary to generate the py file
    const std::string getFQname(const char* id_model) const ;
    std::string generateTableName(std::string FQname) const ;

    std::shared_ptr<StorageInterface> storageInterface; //StorageInterface* storageInterface; /* Connection to Cassandra */


	Writer* dictMetaWriter; /* Writer for dictionary metadata entries in hecuba.istorage */
	CacheTable* numpyMetaAccess; /* Access to hecuba.istorage */
	Writer* numpyMetaWriter; /* CALCULATED: Writer for numpy metadata entries in hecuba.istorage */

    HecubaSession();
    ~HecubaSession();
    void deallocateObjects() ;

    void parse_environment(config_map &config);
    std::vector<std::string> split (std::string s, std::string delimiter) const;
    std::string contact_names_2_IP_addr(std::string &contact_names) const;

    void createSchema(void);
};

#endif /* HECUBA_SESSION_H */
