#include "HecubaSession.h"
#include "ArrayDataStore.h"
#include "SpaceFillingCurve.h"
#include "IStorage.h"
#include "yaml-cpp/yaml.h"
#include "ObjSpec.h"
#include "DataModel.h"

#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <bits/stdc++.h>

#include <iostream>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

//#include "numpy/arrayobject.h" // FIXME to use the numpy constants NPY_*
#define NPY_ARRAY_C_CONTIGUOUS    0x0001
#define NPY_ARRAY_F_CONTIGUOUS    0x0002
#define NPY_ARRAY_OWNDATA         0x0004
#define NPY_ARRAY_FORCECAST       0x0010
#define NPY_ARRAY_ENSURECOPY      0x0020
#define NPY_ARRAY_ENSUREARRAY     0x0040
#define NPY_ARRAY_ELEMENTSTRIDES  0x0080
#define NPY_ARRAY_ALIGNED         0x0100
#define NPY_ARRAY_NOTSWAPPED      0x0200
#define NPY_ARRAY_WRITEABLE       0x0400
#define NPY_ARRAY_UPDATEIFCOPY    0x1000

// This code adds the decoding of YAML files to ObjSpec (NOT the other way around)
namespace YAML {
    template<>
        struct convert<DataModel::datamodel_spec> {
            static Node encode(const DataModel::datamodel_spec& obj) {
                Node node;

                return node;
            }
            static std::string mapUserType(std::string name) {
                /* Translates 'user type' to 'data model type'. numpy.ndarray --> hecuba.hnumpy.StorageNumpy */
                if ((name == "numpy.ndarray") || (name == "StorageNumpy")) {
                    return "hecuba.hnumpy.StorageNumpy";
                }
                name = ObjSpec::yaml_to_cass(name);
                return name;
            }
            static bool decode(const Node& node, DataModel::datamodel_spec& dmodel) {
                ObjSpec::valid_types obj_type;
                std::vector<std::pair<std::string, std::string>> partitionKeys;
                std::vector<std::pair<std::string, std::string>> clusteringKeys;
                std::vector<std::pair<std::string, std::string>> cols;
                std::string pythonString="";

                std::string attrName, attrType;
                if(!node.IsMap()) { return false; }

                const Node typespec = node["TypeSpec"];
                if (!typespec.IsSequence() || (typespec.size() != 2)) { return false; }
                dmodel.id=typespec[0].as<std::string>();
                pythonString="class " + dmodel.id;

                if (typespec[1].as<std::string>() == "StorageDict") {
                    pythonString="from hecuba import StorageDict\n\n" + pythonString;
                    obj_type=ObjSpec::valid_types::STORAGEDICT_TYPE;
                    const Node keyspec =  node["KeySpec"];
                    if (!keyspec.IsSequence() || (keyspec.size() == 0)) { return false; }

                    pythonString+=" (StorageDict):\n   '''\n   @TypeSpec dict <<";


                    if (!keyspec[0].IsSequence() || (keyspec[0].size() != 2)) { return false; }

                    attrName=keyspec[0][0].as<std::string>();
                    attrType=keyspec[0][1].as<std::string>();
                    partitionKeys.push_back(std::pair<std::string,std::string>(attrName,mapUserType(attrType)));
                    pythonString+=attrName+":"+attrType;

                    for (uint32_t i=1; i<keyspec.size();i++) {
                        if (!keyspec[i].IsSequence() || (keyspec[i].size() != 2)) { return false; }

                        attrName=keyspec[i][0].as<std::string>();
                        attrType=keyspec[i][1].as<std::string>();
                        clusteringKeys.push_back(std::pair<std::string,std::string>(attrName,mapUserType(attrType)));
                        pythonString+=","+attrName+":"+attrType;
                    }
                    pythonString+=">";
                    const Node valuespec =  node["ValueSpec"];
                    if (!valuespec.IsSequence() || (valuespec.size() == 0)) { return false; }
                    for (uint32_t i=0; i<valuespec.size();i++) {
                        if (!valuespec[i].IsSequence() || (valuespec[i].size() != 2)) { return false; }
                        attrName=valuespec[i][0].as<std::string>();
                        attrType=valuespec[i][1].as<std::string>();
                        cols.push_back(std::pair<std::string,std::string>(attrName,mapUserType(attrType)));
                        pythonString+=","+attrName+":"+attrType;
                    }
                    pythonString+=">\n   '''";
                } else if (typespec[1].as<std::string>() == "StorageObject") {
                    pythonString="from hecuba import StorageObj\n\n" + pythonString;
                    pythonString=pythonString+" (StorageObj):\n   '''\n";
                    obj_type=ObjSpec::valid_types::STORAGEOBJ_TYPE;
                    const Node classfields =  node["ClassField"];
                    if (!classfields.IsSequence() || (classfields.size() == 0)) { return false; }
                    partitionKeys.push_back(std::pair<std::string,std::string>("storage_id","uuid"));

                    for (uint32_t i=0; i<classfields.size();i++) {
                        if (!classfields[i].IsSequence() || (classfields[i].size() != 2)) { return false; }
                        attrName=classfields[i][0].as<std::string>();
                        attrType=mapUserType(classfields[i][1].as<std::string>());
                        cols.push_back(std::pair<std::string,std::string>(attrName,attrType));
                        pythonString+="   @ClassField "+attrName+" "+attrType+"\n";
                        std::cout<<"7-"<<i<<": "<<pythonString<<std::endl;
                    }
                    pythonString+="   '''\n";
                    std::cout<<"8: "<<pythonString<<std::endl;

                } else return false;

                dmodel.o=ObjSpec(obj_type,partitionKeys,clusteringKeys,cols,pythonString);
                return true;
            }
        };
};

void HecubaSession::parse_environment(config_map &config) {
    const char * nodePort = std::getenv("NODE_PORT");
    if (nodePort == nullptr) {
        nodePort = "9042";
    }
    config["NODE_PORT"] = std::string(nodePort);

    const char * contactNames = std::getenv("CONTACT_NAMES");
    if (contactNames == nullptr) {
        contactNames = "127.0.0.1";
    }
    config["CONTACT_NAMES"] = std::string(contactNames);

    const char * createSchema = std::getenv("CREATE_SCHEMA");
    std::string createSchema2 ;
    if (createSchema == nullptr) {
        createSchema2 = std::string("true");
    } else {
        createSchema2 = std::string(createSchema);
        std::transform(createSchema2.begin(), createSchema2.end(), createSchema2.begin(),
                [](unsigned char c){ return std::tolower(c); });
    }
    config["CREATE_SCHEMA"] = createSchema2;

    const char * executionName = std::getenv("EXECUTION_NAME");
    if (executionName == nullptr) {
        executionName = "my_app";
    }
    config["EXECUTION_NAME"] = std::string(executionName);

    const char * timestampedWrites = std::getenv("TIMESTAMPED_WRITES");
    std::string timestampedWrites2;
    if (timestampedWrites == nullptr) {
        timestampedWrites2 = std::string("false");
    } else {
        timestampedWrites2 = std::string(timestampedWrites);
        std::transform(timestampedWrites2.begin(), timestampedWrites2.end(), timestampedWrites2.begin(),
                [](unsigned char c){ return std::tolower(c); });
    }
    config["TIMESTAMPED_WRITES"] = timestampedWrites2;

        //{"writer_buffer",      std::to_string(writer_queue)},??? == WRITE_BUFFER_SIZE?
    const char * writeBufferSize = std::getenv("WRITE_BUFFER_SIZE");
    if (writeBufferSize == nullptr) {
        writeBufferSize = "1000";
    }
    config["WRITE_BUFFER_SIZE"] = std::string(writeBufferSize);

        ///writer_par ==> 'WRITE_CALLBACKS_NUMBER'
    const char *writeCallbacksNum = std::getenv("WRITE_CALLBACKS_NUMBER");
    if (writeCallbacksNum == nullptr) {
        writeCallbacksNum = "16";
    }
    config["WRITE_CALLBACKS_NUMBER"] = std::string(writeCallbacksNum);

    const char * cacheSize = std::getenv("MAX_CACHE_SIZE");
    if (cacheSize == nullptr) {
        cacheSize = "1000";
    }
    config["MAX_CACHE_SIZE"] = std::string(cacheSize);

    const char *replicationFactor = std::getenv("REPLICA_FACTOR");
    if (replicationFactor == nullptr) {
        replicationFactor = "1";
    }
    config["REPLICA_FACTOR"] = std::string(replicationFactor);

    const char *replicationStrategy = std::getenv("REPLICATION_STRATEGY");
    if (replicationStrategy == nullptr) {
        replicationStrategy = "SimpleStrategy";
    }
    config["REPLICATION_STRATEGY"] = std::string(replicationStrategy);

    const char *replicationStrategyOptions = std::getenv("REPLICATION_STRATEGY_OPTIONS");
    if (replicationStrategyOptions == nullptr) {
        replicationStrategyOptions = "";
    }
    config["REPLICATION_STRATEGY_OPTIONS"] = replicationStrategyOptions;

    if (config["REPLICATION_STRATEGY"] == "SimpleStrategy") {
        config["REPLICATION"] = std::string("{'class' : 'SimpleStrategy', 'replication_factor': ") + config["REPLICA_FACTOR"] + "}";
    } else {
        config["REPLICATION"] = std::string("{'class' : '") + config["REPLICATION_STRATEGY"] + "', " + config["REPLICATION_STRATEGY_OPTIONS"] + "}";
    }
}

CassError HecubaSession::run_query(std::string query) const{
	CassStatement *statement = cass_statement_new(query.c_str(), 0);

    //std::cout << "DEBUG: HecubaSession.run_query : "<<query<<std::endl;
    CassFuture *result_future = cass_session_execute(const_cast<CassSession *>(storageInterface->get_session()), statement);
    cass_statement_free(statement);

    CassError rc = cass_future_error_code(result_future);
    if (rc != CASS_OK) {
        printf("Query execution error: %s - %s\n", cass_error_desc(rc), query.c_str());
    }
    cass_future_free(result_future);
    return rc;
}

void HecubaSession::decodeNumpyMetadata(HecubaSession::NumpyShape *s, void* metadata) {
    // Numpy Metadata(all unsigned): Ndims + Dim1 + Dim2 + ... + DimN  (Layout in C by default)
    unsigned* value = (unsigned*)metadata;
    s->ndims = *(value);
    value ++;
    s->dim = (unsigned *)malloc(s->ndims);
    for (unsigned i = 0; i < s->ndims; i++) {
        s->dim[i] = *(value + i);
    }
}
void HecubaSession::getMetaData(void * raw_numpy_meta, ArrayMetadata &arr_metas) {
    std::vector <uint32_t> dims;
    std::vector <uint32_t> strides;

    // decode void *metadatas
    HecubaSession:: NumpyShape * s = new HecubaSession::NumpyShape();
    decodeNumpyMetadata(s, raw_numpy_meta);
    uint32_t acum=1;
    for (uint32_t i=0; i < s->ndims; i++) {
        dims.push_back( s->dim[i]);
        acum *= s->dim[i];
    }
    for (uint32_t i=0; i < s->ndims; i++) {
        strides.push_back(acum * sizeof(double));
        acum /= s->dim[s->ndims-1-i];
    }
    uint32_t flags=NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED;

    arr_metas.dims = dims;
    arr_metas.strides = strides;
    arr_metas.elem_size = sizeof(double);
    arr_metas.flags = flags;
    arr_metas.partition_type = ZORDER_ALGORITHM;
    arr_metas.typekind = 'f';
    arr_metas.byteorder = '=';
}

void HecubaSession::registerNumpy(ArrayMetadata &numpy_meta, std::string name, uint64_t* uuid) {

    //std::cout<< "DEBUG: HecubaSession::registerNumpy BEGIN "<< name << UUID2str(uuid)<<std::endl;
    void *keys = std::malloc(sizeof(uint64_t *));
    uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);//new uint64_t[2];
    c_uuid[0] = *uuid;
    c_uuid[1] = *(uuid + 1);


    std::memcpy(keys, &c_uuid, sizeof(uint64_t *));


    char *c_name = (char *) std::malloc(name.length() + 1);
    std::memcpy(c_name, name.c_str(), name.length() + 1);

    //COPY VALUES
    int offset = 0;
    uint64_t size_name = strlen(c_name)+1;
    uint64_t size = 0;

    //size of the vector of dims
    size += sizeof(uint32_t) * numpy_meta.dims.size();

    //plus the other metas
    size += sizeof(numpy_meta.elem_size)
		+ sizeof(numpy_meta.partition_type)
		+ sizeof(numpy_meta.flags)
		+ sizeof(uint32_t)*numpy_meta.strides.size() // dims & strides
		+ sizeof(numpy_meta.typekind)
		+ sizeof(numpy_meta.byteorder);

    //allocate plus the bytes counter
    unsigned char *byte_array = (unsigned char *) malloc(size+ sizeof(uint64_t));
    unsigned char *name_array = (unsigned char *) malloc(size_name);


    // copy table name
    memcpy(name_array, c_name, size_name); //lgarrobe

    // Copy num bytes
    memcpy(byte_array+offset, &size, sizeof(uint64_t));
    offset += sizeof(uint64_t);


    //copy everything from the metas
	//	flags int, elem_size int, partition_type tinyint,
    //       dims list<int>, strides list<int>, typekind text, byteorder text


    memcpy(byte_array + offset, &numpy_meta.flags, sizeof(numpy_meta.flags));
    offset += sizeof(numpy_meta.flags);

    memcpy(byte_array + offset, &numpy_meta.elem_size, sizeof(numpy_meta.elem_size));
    offset += sizeof(numpy_meta.elem_size);

    memcpy(byte_array + offset, &numpy_meta.partition_type, sizeof(numpy_meta.partition_type));
    offset += sizeof(numpy_meta.partition_type);

    memcpy(byte_array + offset, &numpy_meta.typekind, sizeof(numpy_meta.typekind));
    offset +=sizeof(numpy_meta.typekind);

    memcpy(byte_array + offset, &numpy_meta.byteorder, sizeof(numpy_meta.byteorder));
    offset +=sizeof(numpy_meta.byteorder);

    memcpy(byte_array + offset, numpy_meta.dims.data(), sizeof(uint32_t) * numpy_meta.dims.size());
    offset +=sizeof(uint32_t)*numpy_meta.dims.size();

    memcpy(byte_array + offset, numpy_meta.strides.data(), sizeof(uint32_t) * numpy_meta.strides.size());
    offset +=sizeof(uint32_t)*numpy_meta.strides.size();

    //memcpy(byte_array + offset, &numpy_meta.inner_type, sizeof(numpy_meta.inner_type));
    //offset += sizeof(numpy_meta.inner_type);

    int offset_values = 0;
    char *values = (char *) malloc(sizeof(char *)*4);

    uint64_t *base_numpy = (uint64_t *) malloc(sizeof(uint64_t) * 2);//new uint64_t[2];
    memcpy(base_numpy, c_uuid, sizeof(uint64_t)*2);
    //std::cout<< "DEBUG: HecubaSession::registerNumpy &base_numpy = "<<base_numpy<<std::endl;
    std::memcpy(values, &base_numpy, sizeof(uint64_t *));  // base_numpy
    offset_values += sizeof(unsigned char *);

    char *class_name=(char*)malloc(strlen("hecuba.hnumpy.StorageNumpy")+1);
    strcpy(class_name, "hecuba.hnumpy.StorageNumpy");
    //std::cout<< "DEBUG: HecubaSession::registerNumpy &class_name = "<<class_name<<std::endl;
    memcpy(values+offset_values, &class_name, sizeof(unsigned char *)); //class_name
    offset_values += sizeof(unsigned char *);

    //std::cout<< "DEBUG: HecubaSession::registerNumpy &name = "<<name_array<<std::endl;
    memcpy(values+offset_values, &name_array, sizeof(unsigned char *)); //name
    offset_values += sizeof(unsigned char *);

    //std::cout<< "DEBUG: HecubaSession::registerNumpy &np_meta = "<<byte_array<<std::endl;
    memcpy(values+offset_values, &byte_array,  sizeof(unsigned char *)); // numpy_meta
    offset_values += sizeof(unsigned char *);

    try {
        numpyMetaWriter->write_to_cassandra(keys, values);
    }
    catch (std::exception &e) {
        std::cerr << "HecubaSession::registerNumpy: Error writing" <<std::endl;
        std::cerr << e.what();
        throw e;
    };

}

void HecubaSession::createSchema(void) {
    // Create hecuba
    std::vector<std::string> queries;
    std::string create_hecuba_keyspace = std::string(
            "CREATE KEYSPACE IF NOT EXISTS hecuba  WITH replication = ") +  config["REPLICATION"];
    queries.push_back(create_hecuba_keyspace);
    std::string create_hecuba_qmeta = std::string(
            "CREATE TYPE IF NOT EXISTS hecuba.q_meta("
            "mem_filter text,"
            "from_point frozen<list<double>>,"
            "to_point frozen<list<double>>,"
            "precision float);");
    queries.push_back(create_hecuba_qmeta);
    std::string create_hecuba_npmeta = std::string(
            "CREATE TYPE IF NOT EXISTS hecuba.np_meta ("
            "flags int, elem_size int, partition_type tinyint,"
            "dims list<int>, strides list<int>, typekind text, byteorder text)");
    queries.push_back(create_hecuba_npmeta);
    std::string create_hecuba_istorage = std::string(
            "CREATE TABLE IF NOT EXISTS hecuba.istorage"
            "(storage_id uuid,"
            "class_name text,name text,"
            "istorage_props map<text,text>,"
            "tokens list<frozen<tuple<bigint,bigint>>>,"
            "indexed_on list<text>,"
            "qbeast_random text,"
            "qbeast_meta frozen<q_meta>,"
            "numpy_meta frozen<np_meta>,"
            "block_id int,"
            "base_numpy uuid,"
            "view_serialization blob,"
            "primary_keys list<frozen<tuple<text,text>>>,"
            "columns list<frozen<tuple<text,text>>>,"
            "PRIMARY KEY(storage_id));");
    queries.push_back(create_hecuba_istorage);
    // Create keyspace EXECUTION_NAME
    std::string create_keyspace = std::string(
            "CREATE KEYSPACE IF NOT EXISTS ") + config["EXECUTION_NAME"] +
        std::string(" WITH replication = ") +  config["REPLICATION"];
    queries.push_back(create_keyspace);

    for(auto q: queries) {
        CassError rc = run_query(q);
        if (rc != CASS_OK) {
            std::string msg = std::string("HecubaSession:: Error Creating Schema executing: ") + q;
            throw ModuleException(msg);
        }
    }
}

/***************************
 * PUBLIC
 ***************************/

/* Constructor: Establish connection with underlying storage system */
HecubaSession::HecubaSession() : currentDataModel(NULL) {

    parse_environment(this->config);



    /* Establish connection */
    this->storageInterface = std::make_shared<StorageInterface>(stoi(config["NODE_PORT"]), config["CONTACT_NAMES"]);
    //this->storageInterface = new StorageInterface(stoi(config["NODE_PORT"]), config["CONTACT_NAMES"]);

    if (this->config["CREATE_SCHEMA"] == "true") {
        createSchema();
    }

// TODO: extend writer to support lists	std::vector<config_map> pkeystypes = { {{"name", "storage_id"}} };
// TODO: extend writer to support lists	std::vector<config_map> ckeystypes = {};
// TODO: extend writer to support lists	std::vector<config_map> colstypes = {{{"name", "class_name"}},
// TODO: extend writer to support lists							{{"name", "name"}},
// TODO: extend writer to support lists							{{"name", "tokens"}},   //list
// TODO: extend writer to support lists							{{"name", "primary_keys"}}, //list
// TODO: extend writer to support lists							{{"name", "columns"}},  //list
// TODO: extend writer to support lists							{{"name", "indexed_on"}} }; //list
// TODO: extend writer to support lists	dictMetaWriter = storageInterface->make_writer("istorage", "hecuba",
// TODO: extend writer to support lists													pkeystypes, colstypes,
// TODO: extend writer to support lists													config);
// TODO: extend writer to support lists


std::vector<config_map> pkeystypes_n = { {{"name", "storage_id"}} };
std::vector<config_map> ckeystypes_n = {};
std::vector<config_map> colstypes_n = {
						{{"name", "base_numpy"}},
                        {{"name", "class_name"}},
						{{"name", "name"}},
						{{"name", "numpy_meta"}},
						//{{"name", "block_id"}}, //NOT USED
						//{{"name", "view_serialization"}},  //Used by Python, uses pickle format. Let it be NULL and initialized at python code
};
						// TODO: extend writer to support lists {{"name", "tokens"}} }; //list

// The attributes stored in istorage for all numpys are the same, we use a single writer for the session
numpyMetaWriter = storageInterface->make_writer("istorage", "hecuba",
												pkeystypes_n, colstypes_n,
												config);

}

HecubaSession::~HecubaSession() {
    delete(currentDataModel);
    delete(numpyMetaWriter);
}

/* loadDataModel: loads a DataModel from 'model_filename' path which should be
 * a YAML file. It also generates its corresponding python generated class in
 * the same directory where the model resides or in the 'pythonSpecPath'
 * directory. */
void HecubaSession::loadDataModel(const char * model_filename, const char * pythonSpecPath=NULL) {

    if (currentDataModel != NULL) {
        std::cerr << "WARNING: HecubaSession::loadDataModel: DataModel already defined. Discarded and load again"<<std::endl;
        delete(currentDataModel);
    }

    // TODO: parse file to get model information NOW HARDCODED

    // class dataModel(StorageDict):
    //    '''
    //         @TypeSpec dict <<lat:int,ts:int>,metrics:numpy.ndarray>
    //    '''


    DataModel* d = new DataModel();


    if (model_filename == NULL) {
		throw std::runtime_error("Trying to load a NULL model file name ");
    }

    // Detect dirname and basename for 'model_filename'
    std::string pythonPath("");
    std::string baseName(model_filename);

    size_t pos = baseName.find_last_of('/');
    if (pos != std::string::npos) { // model_filename has a path
        if (pythonSpecPath != NULL) {
            pythonPath = pythonSpecPath;
        } else {
            pythonPath = baseName.substr(0,pos);
        }
        baseName = baseName.substr(pos+1, baseName.size());
    }
    // Remove .yaml extension
    pos = baseName.find_last_of('.');
    if (pos != std::string::npos) {
        baseName = baseName.substr(0, pos);
    }
    // Create moduleName
    std::string moduleName;
    if (pythonPath == "") {
        moduleName = baseName;
    } else {
        moduleName = pythonPath + "/" + baseName;
    }

    d->setModuleName(moduleName);

    // Add .py extension to moduleName
    moduleName =  moduleName + ".py";

    std::ofstream fd(moduleName);

    YAML::Node node = YAML::LoadFile(model_filename);

    assert(node.Type() == YAML::NodeType::Sequence);

    for(std::size_t i=0;i<node.size();i++) {
		DataModel::datamodel_spec x = node[i].as<DataModel::datamodel_spec>();
		d->addObjSpec(x.id, x.o);
        fd<<x.o.getPythonString();
    }
    fd.close();

    // ALWAYS Add numpy (just in case) ##############################
    std::vector<std::pair<std::string, std::string>> pkeystypes_numpy = {
                                  {"storage_id", "uuid"},
                                  {"cluster_id", "int"}
    };
    std::vector<std::pair<std::string, std::string>> ckeystypes_numpy = {{"block_id","int"}};
    std::vector<std::pair<std::string, std::string>> colstypes_numpy = {
                                  {"payload", "blob"},
    };
    d->addObjSpec(ObjSpec::valid_types::STORAGENUMPY_TYPE, "hecuba.hnumpy.StorageNumpy", pkeystypes_numpy, ckeystypes_numpy, colstypes_numpy);
    // ##############################

    currentDataModel = d;
}

IStorage* HecubaSession::createObject(const char * id_model, const char * id_object, void * metadata, void* value) {
    // Create Cassandra tables 'ksp.id_object' for object 'id_object' according to its type 'id_model' in 'model'
    // TODO: create type depending on 'model', now its only dictionary

    DataModel* model = currentDataModel;
    if (model == NULL) {
        throw ModuleException("HecubaSession::createObject No data model loaded");
    }

    IStorage * o;

    ObjSpec oType = model->getObjSpec(id_model);
    //std::cout << "DEBUG: HecubaSession::createObject '"<<id_model<< "' ==> " <<oType.debug()<<std::endl;

    std::string object_name(config["EXECUTION_NAME"] + "." + std::string(id_object));
    uint64_t *c_uuid = generateUUID5(object_name.c_str()); // UUID for the new object

    switch(oType.getType()) {
        case ObjSpec::valid_types::STORAGEOBJ_TYPE:
            {
                // StorageObj case
                //  Create table 'class_name' "CREATE TABLE ksp.class_name (storage_id UUID, nom typ, ... PRIMARY KEY (storage_id))"
                std::string query = "CREATE TABLE IF NOT EXISTS " +
                    config["EXECUTION_NAME"] + "." + id_model +
                    oType.table_attr;

                CassError rc = run_query(query);
                if (rc != CASS_OK) {
                    if (rc == CASS_ERROR_SERVER_INVALID_QUERY) { // keyspace does not exist
                        std::cout<< "HecubaSession Creating keyspace "<< config["EXECUTION_NAME"]<< std::endl;
                        std::string create_keyspace = std::string(
                                "CREATE KEYSPACE IF NOT EXISTS ") + config["EXECUTION_NAME"] +
                            std::string(" WITH replication = ") +  config["REPLICATION"];
                        rc = run_query(create_keyspace);
                        if (rc != CASS_OK) {
                            std::string msg = std::string("HecubaSession:: Error executing query ") + create_keyspace;
                            throw ModuleException(msg);
                        } else {
                            rc = run_query(query); // Repeat table creation after creating keyspace
                            if (rc != CASS_OK) {
                                std::string msg = std::string("HecubaSession:: Error executing query ") + query;
                                throw ModuleException(msg);
                            }
                        }
                    } else {
                        std::string msg = std::string("HecubaSession:: Error executing query ") + query;
                        throw ModuleException(msg);
                    }
                }
                // Table for storageobj class created
                // Add entry to ISTORAGE: TODO add the tokens attribute
                std::string name = config["EXECUTION_NAME"] + "." + id_object;
                std::string insquery = std::string("INSERT INTO ") +
                    std::string("hecuba.istorage") +
                    std::string("(storage_id, name, class_name, columns)") +
                    std::string("VALUES ") +
                    std::string("(") +
                    UUID2str(c_uuid) + std::string(", ") +
                    "'" + name + "'" + std::string(", ") +
                    "'" + model->getModuleName() + "." + id_model + "'" + std::string(", ") +
                    oType.getColsStr() +
                    std::string(")");
                run_query(insquery);

                //  Create Writer for storageobj
                std::vector<config_map>* keyNamesDict = oType.getKeysNamesDict();
                std::vector<config_map>* colNamesDict = oType.getColsNamesDict();

                Writer *writer = storageInterface->make_writer(id_model, config["EXECUTION_NAME"].c_str(),
                          *keyNamesDict, *colNamesDict,
                          config);
                delete keyNamesDict;
                delete colNamesDict;
                o = new IStorage(this, id_model, config["EXECUTION_NAME"] + "." + id_object, c_uuid, writer);

            }
            break;
        case ObjSpec::valid_types::STORAGEDICT_TYPE:
            {
                // Dictionary case
                //  Create table 'name' "CREATE TABLE ksp.name (nom typ, nom typ, ... PRIMARY KEY (nom, nom))"
                bool new_element = true;
                std::string query = "CREATE TABLE " +
                    config["EXECUTION_NAME"] + "." + id_object +
                    oType.table_attr;

                CassError rc = run_query(query);
                if (rc != CASS_OK) {
                    if (rc == CASS_ERROR_SERVER_ALREADY_EXISTS ) {
                        new_element = false; //OOpps, creation failed. It is an already existent object.
                    } else if (rc == CASS_ERROR_SERVER_INVALID_QUERY) {
                        std::cout<< "HecubaSession Creating keyspace "<< config["EXECUTION_NAME"]<< std::endl;
                        std::string create_keyspace = std::string(
                                "CREATE KEYSPACE IF NOT EXISTS ") + config["EXECUTION_NAME"] +
                            std::string(" WITH replication = ") +  config["REPLICATION"];
                        rc = run_query(create_keyspace);
                        if (rc != CASS_OK) {
                            std::string msg = std::string("HecubaSession:: Error executing query ") + create_keyspace;
                            throw ModuleException(msg);
                        } else {
                            rc = run_query(query);
                            if (rc != CASS_OK) {
                                if (rc == CASS_ERROR_SERVER_ALREADY_EXISTS) {
                                    new_element = false; //OOpps, creation failed. It is an already existent object.
                                }  else {
                                    std::string msg = std::string("HecubaSession:: Error executing query ") + query;
                                    throw ModuleException(msg);
                                }
                            }
                        }
                    } else {
                        std::string msg = std::string("HecubaSession:: Error executing query ") + query;
                        throw ModuleException(msg);
                    }
                }

                if (new_element) {
                    //  Add entry to hecuba.istorage: TODO add the tokens attribute
                    std::string name = config["EXECUTION_NAME"] + "." + id_object;
                    // TODO EXPECTED:vvv NOW HARDCODED
                    //classname = id_model
                    // keys = {c_uuid}, values={name, class_name, primary_keys, columns } # no tokens, no numpy_meta, ...
                    //try {
                    //	dictMetaWriter->write_to_cassandra(keys, values);
                    //}
                    //catch (std::exception &e) {
                    //	std::cerr << "Error writing in registering" <<std::endl;
                    //	std::cerr << e.what();
                    //	throw e;
                    //}
                    // EXPECTED^^^^
                    // Example: insert into hecuba.istorage (storage_id, primary_keys) values (3dd30d5d-b0d4-45b5-a21a-c6ad313007fd, [('lat','int'),('ts','int')]);
                    std::string insquery = std::string("INSERT INTO ") +
                        std::string("hecuba.istorage") +
                        std::string("(storage_id, name, class_name, primary_keys, columns)") +
                        std::string("VALUES ") +
                        std::string("(") +
                        UUID2str(c_uuid) + std::string(", ") +
                        "'" + name + "'" + std::string(", ") +
                        "'" + id_model + "'" + std::string(", ") +
                        oType.getKeysStr() + std::string(", ") +
                        oType.getColsStr() +
                        std::string(")");
                    run_query(insquery);
                } else {
                    std::cerr << "WARNING: Object "<<id_object<<" already exists. Trying to overwrite it. It may fail if the schema does not match."<<std::endl;
                    // TODO: THIS IS NOT WORKING. We need to get the storage_id (c_uuid) from istorage DISABLE
                    // TODO: Check the schema in Cassandra matches the model
                }

                //  Create Writer for dictionary
                std::vector<config_map>* keyNamesDict = oType.getKeysNamesDict();
                std::vector<config_map>* colNamesDict = oType.getColsNamesDict();


                Writer *writer = storageInterface->make_writer(id_object, config["EXECUTION_NAME"].c_str(),
                          *keyNamesDict, *colNamesDict,
                          config);
                delete keyNamesDict;
                delete colNamesDict;
                o = new IStorage(this, id_model, config["EXECUTION_NAME"] + "." + id_object, c_uuid, writer);
            }
            break;

        case ObjSpec::valid_types::STORAGENUMPY_TYPE:
            {
                // Create table
                std::string query = "CREATE TABLE IF NOT EXISTS " + config["EXECUTION_NAME"] + "." + id_object +
                    " (storage_id uuid, cluster_id int, block_id int, payload blob, "
                    "PRIMARY KEY((storage_id,cluster_id),block_id)) "
                    "WITH compaction = {'class': 'SizeTieredCompactionStrategy', 'enabled': false};";

                this->run_query(query);

                // StorageNumpy
                ArrayDataStore *array_store = new ArrayDataStore(id_object, config["EXECUTION_NAME"].c_str(),
                        this->storageInterface->get_session(), config);
                //std::cout << "DEBUG: HecubaSession::createObject After ArrayDataStore creation " <<std::endl;

                // Create entry in hecuba.istorage for the new numpy
                ArrayMetadata numpy_metas;
                getMetaData(metadata, numpy_metas); // numpy_metas = getMetaData(metadata);
                //std::cout<< "DEBUG: HecubaSession::createObject After metadata creation " <<std::endl;
                registerNumpy(numpy_metas, id_object, c_uuid);
                //std::cout<< "DEBUG: HecubaSession::createObject After REGISTER numpy into ISTORAGE" <<std::endl;

                //Create keys, values to store the numpy
                //double* tmp = (double*)value;
                //std::cout<< "DEBUG: HecubaSession::createObject BEFORE store FIRST element in NUMPY is "<< *tmp << " and second is "<<*(tmp+1)<< " 3rd " << *(tmp+2)<< std::endl;
                array_store->store_numpy_into_cas(c_uuid, numpy_metas, value);
                array_store->wait_stores();

                // TODO: el writer para pasarselo al istorage esta en la cache table del array datastor: anaydir getCache al arraydatastore
                o = new IStorage(this, id_model, config["EXECUTION_NAME"] + "." + id_object, c_uuid, array_store->getWriteCache()->get_writer());

            }
            break;
        default:
            throw ModuleException("HECUBA Session: createObject Unknown type ");// + std::string(oType.objtype));
            break;
    }
    //std::cout << "DEBUG: HecubaSession::createObject DONE" << std::endl;
    return o;
}

uint64_t* HecubaSession::generateUUID(void) const {
    uint64_t *c_uuid; // UUID for the new object
    c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);

    boost::uuids::random_generator gen;
    boost::uuids::uuid u = gen();

    memcpy(c_uuid, &u, 16);

    return c_uuid;
}

uint64_t* HecubaSession::generateUUID5(const char* name) const{
    /* uses sha1 */
    uint64_t *c_uuid; // UUID for the new object
    c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);

    boost::uuids::string_generator gen1;
    boost::uuids::uuid dns_namespace_uuid = gen1("{6ba7b810-9dad-11d1-80b4-00c04fd430c8}");
    boost::uuids::name_generator gen(dns_namespace_uuid);

    // boost::uuids::name_generator_md5 gen(boost::uuids::ns::dns());

    boost::uuids::uuid u = gen(name);

    memcpy(c_uuid, &u, 16);

    return c_uuid;
}

std::string HecubaSession::UUID2str(uint64_t* c_uuid) {
    /* This MUST match with the 'cass_statement_bind_uuid' result */
    char str[37] = {};
    unsigned char* uuid = reinterpret_cast<unsigned char*>(c_uuid);
    //std::cout<< "HecubaSession: uuid2str: BEGIN "<<std::hex<<c_uuid[0]<<c_uuid[1]<<std::endl;
    sprintf(str,
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        uuid[0], uuid[1], uuid[2], uuid[3],
        uuid[4], uuid[5],
        uuid[6], uuid[7],
        uuid[8], uuid[9],
        uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]
        );
    //std::cout<< "HecubaSession: uuid2str: "<<str<<std::endl;
    return std::string(str);
}

DataModel* HecubaSession::getDataModel() {
    return currentDataModel;
}
