#ifndef HECUBA_SESSION_H
#define HECUBA_SESSION_H

#include <fstream>
#include <iostream>
#include "StorageInterface.h"
//#include "MetaManager.h"
#include "configmap.h"
#include "DataModel.h"
#include "ObjSpec.h"

class IStorage;  //Forward Declaration

class HecubaSession {
public:
    /** Establish connection with Underlying storage system */
    HecubaSession();
    ~HecubaSession();

    void registerObject(IStorage * d);
    void loadDataModel(const char * model_filename, const char *python_spec_path=nullptr);
    DataModel* getDataModel();

	struct NumpyShape {
		unsigned ndims; //Number of dimensions
		unsigned* dim;  //Dimensions

        std::string debug() {
            std::string res="";
            for(unsigned d=0; d < ndims; d++) {
                res += std::to_string(dim[d]);
                if (d != ndims - 1) { res += ","; }
            }
            return res;
        }
	};

    /* createObject : Creates a new object of class 'id_model' with name 'id_object'
     * Returns: A new IStorage reference. User MUST delete this reference after use */
    IStorage* createObject(const char * id_model, const char * id_object, void* metadata=NULL, void* value=NULL); //Special case to set a Numpy

    /* createObject : Instantiate an existing object of class 'id_model' with id 'uuid'
     * Returns: A new IStorage reference. User MUST delete this reference after use */
    IStorage* createObject(const char * id_model, uint64_t* uuid);


    //Writer* getDictMetaWriter();
    //Writer* getNumpyMetaWriter();

    //config_map getConfig();

    config_map config;
    std::shared_ptr<StorageInterface> getStorageInterface() {
        return storageInterface;
    }; /* Connection to Cassandra */
    std::string getExecutionName();
    CassError run_query(std::string) const;

    Writer * getNumpyMetaWriter() const;
    CacheTable * getHecubaIstorageAccess() const;
private:

    void decodeNumpyMetadata(HecubaSession::NumpyShape *s, void* metadata);
    const std::string getFQname(const char* id_model) const ;
    std::string generateTableName(std::string FQname) const ;

    std::shared_ptr<StorageInterface> storageInterface; //StorageInterface* storageInterface; /* Connection to Cassandra */

    DataModel* currentDataModel; /* loaded Data Model */

	Writer* dictMetaWriter; /* Writer for dictionary metadata entries in hecuba.istorage */
	CacheTable* numpyMetaAccess; /* Access to hecuba.istorage */
	Writer* numpyMetaWriter; /* CALCULATED: Writer for numpy metadata entries in hecuba.istorage */

    //MetaManager mm; //* To be deleted? */

    void parse_environment(config_map &config);
    void getMetaData(void* raw_numpy_meta, ArrayMetadata &arr_metas);
    void registerNumpy(ArrayMetadata &numpy_meta, std::string name, uint64_t* uuid);
    std::vector<std::string> split (std::string s, std::string delimiter) const;
    std::string contact_names_2_IP_addr(std::string &contact_names) const;

    void createSchema(void);
};

#endif /* HECUBA_SESSION_H */
