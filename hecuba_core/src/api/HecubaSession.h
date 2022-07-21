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

    void loadDataModel(const char * model_filename, const char *python_spec_path);
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

    IStorage* createObject(const char * id_model, const char * id_object, void* metadata=NULL, void* value=NULL); //Special case to set a Numpy

    //Writer* getDictMetaWriter();
    //Writer* getNumpyMetaWriter();

    //config_map getConfig();

private:

    void decodeNumpyMetadata(HecubaSession::NumpyShape *s, void* metadata);

    std::shared_ptr<StorageInterface> storageInterface; //StorageInterface* storageInterface; /* Connection to Cassandra */

    DataModel* currentDataModel; /* loaded Data Model */

	Writer* dictMetaWriter; /* Writer for dictionary metadata entries in hecuba.istorage */
	Writer* numpyMetaWriter; /* Writer for numpy metadata entries in hecuba.istorage */

    //MetaManager mm; //* To be deleted? */
    config_map config;

    void parse_environment(config_map &config);
	CassError run_query(std::string) const;
    void getMetaData(void* raw_numpy_meta, ArrayMetadata &arr_metas);
    void registerNumpy(ArrayMetadata &numpy_meta, std::string name, uint64_t* uuid);

    void createSchema(void);
};

#endif /* HECUBA_SESSION_H */
