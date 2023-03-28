#include "HecubaSession.h"
#include "ArrayDataStore.h"
#include "SpaceFillingCurve.h"
#include "IStorage.h"
#include "ObjSpec.h"
#include "UUID.h"

#include "debug.h"

#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <bits/stdc++.h>
#include <cxxabi.h>

#include <iostream>

 #include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <typeinfo>


std::vector<std::string> HecubaSession::split (std::string s, std::string delimiter) const{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

/** contact_names_2_IP_addr: Given a string with a list of comma separated of
 * hostnames, returns a string with same hosts as IP address */
std::string HecubaSession::contact_names_2_IP_addr(std::string &contact_names)
const {
    std::vector<std::string> contact;
    std::vector<std::string> contact_ips;

    struct addrinfo hints;
    struct addrinfo *result;

    // Split contact_names
    contact = split(contact_names, ",");
    if (contact.size()==0) {
        fprintf(stderr, "Empty contact_names ");
        return std::string("");
    }


    /* Obtain address(es) matching host/port */

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
    hints.ai_socktype = SOCK_DGRAM; /* Datagram socket */
    hints.ai_flags = 0;
    hints.ai_protocol = 0;          /* Any protocol */


    for (uint32_t i=0; i<contact.size(); i++) {
        int s = getaddrinfo(contact[i].c_str(), NULL, &hints, &result);
        if (s != 0) {
            fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
            return std::string("");
        }
        /* getaddrinfo() returns a list of address structures.
           Try each address until we successfully connect(2).
           If socket(2) (or connect(2)) fails, we (close the socket
           and) try the next address. */

        if (result == NULL) {               /* No address succeeded */
            std::cerr<<"Address "<<contact[i]<<" is invalid\n"<<std::endl;
            return std::string("");
        }

        char host[NI_MAXHOST];
        if (getnameinfo(result->ai_addr, result->ai_addrlen
                    , host, sizeof(host)
                    , NULL, 0
                    , NI_NUMERICHOST) != 0) {
            std::cerr<<"Address "<<contact[i]<<" unable to get IP address: "<<strerror(errno)<<std::endl;
            return std::string("");
        }
        DBG("Address "<<contact[i]<<" translated to " <<std::string(host));
        contact_ips.push_back(host);

        freeaddrinfo(result);           /* No longer needed */

    }
    std::string contactips_str(contact_ips[0]);
    for (uint32_t i=1; i<contact_ips.size(); i++) {
        contactips_str+= "," + contact_ips[i];
    }
    return contactips_str;
}

void HecubaSession::parse_environment(config_map &config) {
    const char * nodePort = std::getenv("NODE_PORT");
    if (nodePort == nullptr) {
        nodePort = "9042";
    }
    config["node_port"] = std::string(nodePort);

    const char * contactNames = std::getenv("CONTACT_NAMES");
    if (contactNames == nullptr) {
        contactNames = "127.0.0.1";
    }
    // Transform Names to IP addresses (Cassandra's fault: cassandra_query_set_host needs an IP number)
    std::string cnames = std::string(contactNames);
    config["contact_names"] = contact_names_2_IP_addr(cnames);



    const char * kafkaNames = std::getenv("KAFKA_NAMES");
    if (kafkaNames == nullptr) {
        kafkaNames = contactNames;
    }
    config["kafka_names"] = std::string(kafkaNames);

    const char * createSchema = std::getenv("CREATE_SCHEMA");
    std::string createSchema2 ;
    if (createSchema == nullptr) {
        createSchema2 = std::string("true");
    } else {
        createSchema2 = std::string(createSchema);
        std::transform(createSchema2.begin(), createSchema2.end(), createSchema2.begin(),
                [](unsigned char c){ return std::tolower(c); });
    }
    config["create_schema"] = createSchema2;

    const char * executionName = std::getenv("EXECUTION_NAME");
    if (executionName == nullptr) {
        executionName = "my_app";
    }
    config["execution_name"] = std::string(executionName);

    const char * timestampedWrites = std::getenv("TIMESTAMPED_WRITES");
    std::string timestampedWrites2;
    if (timestampedWrites == nullptr) {
        timestampedWrites2 = std::string("false");
    } else {
        timestampedWrites2 = std::string(timestampedWrites);
        std::transform(timestampedWrites2.begin(), timestampedWrites2.end(), timestampedWrites2.begin(),
                [](unsigned char c){ return std::tolower(c); });
    }
    config["timestamped_writes"] = timestampedWrites2;

        //{"writer_buffer",      std::to_string(writer_queue)},??? == WRITE_BUFFER_SIZE?
    const char * writeBufferSize = std::getenv("WRITE_BUFFER_SIZE");
    if (writeBufferSize == nullptr) {
        writeBufferSize = "1000";
    }
    config["write_buffer_size"] = std::string(writeBufferSize);

        ///writer_par ==> 'WRITE_CALLBACKS_NUMBER'
    const char *writeCallbacksNum = std::getenv("WRITE_CALLBACKS_NUMBER");
    if (writeCallbacksNum == nullptr) {
        writeCallbacksNum = "16";
    }
    config["write_callbacks_number"] = std::string(writeCallbacksNum);

    const char * cacheSize = std::getenv("MAX_CACHE_SIZE");
    if (cacheSize == nullptr) {
        cacheSize = "1000";
    }
    config["max_cache_size"] = std::string(cacheSize);

    const char *replicationFactor = std::getenv("REPLICA_FACTOR");
    if (replicationFactor == nullptr) {
        replicationFactor = "1";
    }
    config["replica_factor"] = std::string(replicationFactor);

    const char *replicationStrategy = std::getenv("REPLICATION_STRATEGY");
    if (replicationStrategy == nullptr) {
        replicationStrategy = "SimpleStrategy";
    }
    config["replication_strategy"] = std::string(replicationStrategy);

    const char *replicationStrategyOptions = std::getenv("REPLICATION_STRATEGY_OPTIONS");
    if (replicationStrategyOptions == nullptr) {
        replicationStrategyOptions = "";
    }
    config["replication_strategy_options"] = replicationStrategyOptions;

    if (config["replication_strategy"] == "SimpleStrategy") {
        config["replication"] = std::string("{'class' : 'SimpleStrategy', 'replication_factor': ") + config["replica_factor"] + "}";
    } else {
        config["replication"] = std::string("{'class' : '") + config["replication_strategy"] + "', " + config["replication_strategy_options"] + "}";
    }

    const char *hecubaSNSingleTable = std::getenv("HECUBA_SN_SINGLE_TABLE");
    std::string hecubaSNSingleTable2;
    if (hecubaSNSingleTable == nullptr) { // Default
        hecubaSNSingleTable2 = std::string("true");
    } else {
        hecubaSNSingleTable2 = std::string(hecubaSNSingleTable);
        std::transform(hecubaSNSingleTable2.begin(), hecubaSNSingleTable2.end(), hecubaSNSingleTable2.begin(),
                [](unsigned char c){ return std::tolower(c); });
    }
    config["hecuba_sn_single_table"] = hecubaSNSingleTable2;
}

CassError HecubaSession::run_query(std::string query) const{
	CassStatement *statement = cass_statement_new(query.c_str(), 0);

    //std::cout << "DEBUG: HecubaSession.run_query : "<<query<<std::endl;
    CassFuture *result_future = cass_session_execute(const_cast<CassSession *>(storageInterface->get_session()), statement);
    cass_statement_free(statement);

    CassError rc = cass_future_error_code(result_future);
    if (rc != CASS_OK) {
        //printf("Query execution error: %s - %s\n", cass_error_desc(rc), query.c_str());
    }
    cass_future_free(result_future);
    return rc;
}

Writer * HecubaSession::getNumpyMetaWriter() const {
    return numpyMetaWriter;
}

CacheTable * HecubaSession::getHecubaIstorageAccess() const {
    return numpyMetaAccess;
}

void HecubaSession::createSchema(void) {
    // Create hecuba
    std::vector<std::string> queries;
    std::string create_hecuba_keyspace = std::string(
            "CREATE KEYSPACE IF NOT EXISTS hecuba  WITH replication = ") +  config["replication"];
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
            "CREATE KEYSPACE IF NOT EXISTS ") + config["execution_name"] +
        std::string(" WITH replication = ") +  config["replication"];
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
HecubaSession& HecubaSession::get() {
    static HecubaSession currentSession;
	std::cout << " DEBUG : HecubaSession::get() [ "<< &currentSession << " ] "<<std::endl;
    return currentSession;
}


/* Constructor: Establish connection with underlying storage system */
HecubaSession::HecubaSession() {

    parse_environment(this->config);



    /* Establish connection */
    this->storageInterface = std::make_shared<StorageInterface>(stoi(config["node_port"]), config["contact_names"]);
    //this->storageInterface = new StorageInterface(stoi(config["node_port"]), config["contact_names"]);

    if (this->config["create_schema"] == "true") {
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
numpyMetaAccess = storageInterface->make_cache("istorage", "hecuba",
												pkeystypes_n, colstypes_n,
												config);
numpyMetaWriter = numpyMetaAccess->get_writer();


}

HecubaSession::~HecubaSession() {
    delete(numpyMetaAccess);
    for (std::list<std::shared_ptr<CacheTable>>::iterator it = alive_objects.begin(); it != alive_objects.end();) {
        std::shared_ptr<CacheTable> t = *it;
        std::cout << "LIST DEL: "<< t.get() <<" ("<<t.use_count()<<")"<<std::endl;
        it = alive_objects.erase(it); // this will block waiting for the 'sync'
    }
    for (std::list<std::shared_ptr<ArrayDataStore>>::iterator it = alive_numpy_objects.begin(); it != alive_numpy_objects.end();) {
        std::shared_ptr<ArrayDataStore> t = *it;
        std::cout << "LIST DEL: "<< t.get() <<" ("<<t.use_count()<<")"<<std::endl;
        it = alive_numpy_objects.erase(it); // this will block waiting for the 'sync'
    }
}

/* Given a class name 'id_model' returns its Fully Qualified Name with Python
 * Style using the current Data Model modulename.
 * Examples:
 *      classname --> classname.classname
 *      hecuba.hnumpy.StorageNumpy --> hecuba.hnumpy.StorageNumpy
 *      path1.path2.classname -> NOT SUPPORTED YET (should be the same)
 */
const std::string HecubaSession::getFQname(const char* id_model) const {
    std::string FQid_model (id_model);
    if (strcmp(id_model, "hecuba.hnumpy.StorageNumpy")==0) {
        // Special treatment for NUMPY
        FQid_model = "hecuba.hnumpy.StorageNumpy";

    } else if (FQid_model.find_first_of(".") ==  std::string::npos) {
        // FQid_model: Fully Qualified name for the id_model: module_name.id_model
        //      In YAML we allow to define the class_name without the model:
        //          file: model_complex.py
        //             class info (StorageObj):
        //                  ...
        //      But we store the Fully Qualified name> "model_complex.info"
        FQid_model.insert(0, std::string(id_model) + ".");

    }
    return FQid_model;
}

/* Given a FQname return a name suitable to be stored as a tablename in Cassandra */
std::string HecubaSession::generateTableName(std::string FQname) const {
    // FIXME: We currently only allow classes from a unique
    // model, because just the class name is stored in cassandra
    // without any reference to the modulename. An option could be
    // to store the modulename and the classname separated by '_'.
    // For now, we just keep the classname as tablename
    std::string table_name (FQname);
    int pos = table_name.find_last_of(".");
    table_name = table_name.substr(pos+1);
    return table_name;
}

#if 0
IStorage* HecubaSession::createObject(const char * id_model, uint64_t* uuid) {
    // Instantitate an existing object

    DataModel* model = currentDataModel;
    if (model == NULL) {
        throw ModuleException("HecubaSession::createObject No data model loaded");
    }

    // FQid_model: Fully Qualified name for the id_model: module_name.id_model
    //      In YAML we allow to define the class_name without the model:
    //          file: model_complex.py
    //             class info (StorageObj):
    //                  ...
    //      But we store the Fully Qualified name> "model_complex.info"
    std::string FQid_model = getFQname(id_model);

    IStorage * o;
    ObjSpec oType = model->getObjSpec(FQid_model);
    DBG(" HecubaSession::createObject INSTANTIATING " << oType.debug());
    switch(oType.getType()) {
        case ObjSpec::valid_types::STORAGEOBJ_TYPE:
        case ObjSpec::valid_types::STORAGEDICT_TYPE:
            {
                // read from istorage: uuid --> id_object (name)
                // A new buffer for the uuid (key) is needed (Remember that
                // retrieve_from_cassandra creates a TupleRow of the parameter
                // and therefore the parameter can NOT be a stack pointer... as
                // it will be freed on success)
                void * localuuid = malloc(2*sizeof(uint64_t));
                memcpy(localuuid, uuid, 2*sizeof(uint64_t));
                void * key = malloc(sizeof(char*));
                memcpy(key, &localuuid, sizeof(uint64_t*));

                std::vector <const TupleRow*> result = numpyMetaAccess->retrieve_from_cassandra(key);

                if (result.empty()) throw ModuleException("HecubaSession::createObject uuid "+UUID::UUID2str(uuid)+" not found. Unable to instantiate");

                uint32_t pos = numpyMetaAccess->get_metadata()->get_columnname_position("name");
                char *keytable = *(char**)result[0]->get_element(pos); //Value retrieved from cassandra has 'keyspace.tablename' format

                std::string keyspace (keytable);
                std::string tablename;
                pos = keyspace.find_first_of('.');
                tablename = keyspace.substr(pos+1);
                keyspace = keyspace.substr(0,pos);

                const char * id_object = tablename.c_str();

                // Check that retrieved classname form hecuba coincides with 'id_model'
                pos = numpyMetaAccess->get_metadata()->get_columnname_position("class_name");
                char *classname = *(char**)result[0]->get_element(pos); //Value retrieved from cassandra has 'keyspace.tablename' format
                std::string sobj_table_name (classname);

                // The class_name retrieved in the case of the storageobj is
                // the fully qualified name, but in cassandra the instances are
                // stored in a table with the name of the last part(example:
                // "model_complex.info" will have instances in "keyspace.info")
                // meaning that in a complex scenario with different models...
                // we will loose information. FIXME
                if (sobj_table_name.compare(FQid_model) != 0) {
                    throw ModuleException("HecubaSession::createObject uuid "+UUID::UUID2str(uuid)+" "+ keytable + " has unexpected class_name " + sobj_table_name + " instead of "+FQid_model);
                }



                //  Create Writer for storageobj
                std::vector<config_map>* keyNamesDict = oType.getKeysNamesDict();
                std::vector<config_map>* colNamesDict = oType.getColsNamesDict();

                CacheTable *dataAccess = NULL;
                if (oType.getType() == ObjSpec::valid_types::STORAGEOBJ_TYPE) {
                    dataAccess = storageInterface->make_cache(generateTableName(FQid_model).c_str(), keyspace.c_str(),
                            *keyNamesDict, *colNamesDict,
                            config);
                } else {
                    dataAccess = storageInterface->make_cache(id_object, keyspace.c_str(),
                            *keyNamesDict, *colNamesDict,
                            config);
                }
                delete keyNamesDict;
                delete colNamesDict;

                // IStorage needs a UUID pointer... but the parameter 'uuid' is
                // from the user, therefore we can not count on it
                localuuid = malloc(2*sizeof(uint64_t));
                memcpy(localuuid, uuid, 2*sizeof(uint64_t));

                o = new IStorage(this, FQid_model, keyspace + "." + id_object, (uint64_t*)localuuid, dataAccess);

                if (oType.isStream()) {
                    std::string topic = std::string(UUID::UUID2str(uuid));
                    DBG("     AND IT IS AN STREAM!");
                    o->configureStream(topic);
                }
            }
            break;
        case ObjSpec::valid_types::STORAGENUMPY_TYPE:
            {
                // read from istorage: uuid --> metadata and id_object
                // A new buffer for the uuid (key) is needed (Remember that
                // retrieve_from_cassandra creates a TupleRow of the parameter
                // and therefore the parameter can NOT be a stack pointer... as
                // it will be freed on success)


#if 0
                void * localuuid = malloc(2*sizeof(uint64_t));
                memcpy(localuuid, uuid, 2*sizeof(uint64_t));
                void * key = malloc(sizeof(char*));
                memcpy(key, &localuuid, sizeof(uint64_t*));

                std::vector <const TupleRow*> result = numpyMetaAccess->retrieve_from_cassandra(key);

                if (result.empty()) throw ModuleException("HecubaSession::createObject uuid "+UUID::UUID2str(uuid)+" not found. Unable to instantiate");

                uint32_t pos = numpyMetaAccess->get_metadata()->get_columnname_position("name");
                char *keytable = *(char**)result[0]->get_element(pos); //Value retrieved from cassandra has 'keyspace.tablename' format

                std::string keyspace (keytable);
                std::string tablename;
                pos = keyspace.find_first_of('.');
                tablename = keyspace.substr(pos+1);
                keyspace = keyspace.substr(0,pos);

                // Read the UDT case (numpy_meta)from the row retrieved from cassandra
                pos = numpyMetaAccess->get_metadata()->get_columnname_position("numpy_meta");
                ArrayMetadata *numpy_metas = *(ArrayMetadata**)result[0]->get_element(pos);
                DBG("DEBUG: HecubaSession::createNumpy . Size "<< numpy_metas->get_array_size());

                // StorageNumpy
                ArrayDataStore *array_store = new ArrayDataStore(tablename.c_str(), keyspace.c_str(),
                        this->storageInterface->get_session(), config);
                //std::cout << "DEBUG: HecubaSession::createObject After ArrayDataStore creation " <<std::endl;

                // IStorage needs a UUID pointer... but the parameter 'uuid' is
                // from the user, therefore we can not count on it
                localuuid = malloc(2*sizeof(uint64_t));
                memcpy(localuuid, uuid, 2*sizeof(uint64_t));

                o = new IStorage(this, FQid_model, keytable, (uint64_t*)localuuid, array_store->getWriteCache());
                o->setNumpyAttributes(array_store, *numpy_metas); // SET METAS and DATA!!
                if (oType.isStream()) {
                    std::string topic = std::string(UUID::UUID2str(uuid));
                    DBG("     AND IT IS AN STREAM!");
                    o->configureStream(topic);
                }
#else
		throw ModuleException("HECUBA Session: createObject for already created numpy NOT IMPLEMENTED");
#endif
            }
            break;
        default:
            throw ModuleException("HECUBA Session: createObject Unknown type ");// + std::string(oType.objtype));
            break;
    }
    return o;
}

IStorage* HecubaSession::createObject(const char * id_model, const char * id_object, void * metadata, void* value) {
    // Create Cassandra tables 'ksp.id_object' for object 'id_object' according to its type 'id_model' in 'model'

    DataModel* model = currentDataModel;
    if (model == NULL) {
        throw ModuleException("HecubaSession::createObject No data model loaded");
    }

    std::string FQid_model = getFQname(id_model);

    IStorage * o;
    ObjSpec oType = model->getObjSpec(FQid_model);
    //std::cout << "DEBUG: HecubaSession::createObject '"<<FQid_model<< "' ==> " <<oType.debug()<<std::endl;

    std::string id_object_str;
    if (id_object == nullptr) { //No name used, generate a new one
        id_object_str = "X" + UUID::UUID2str(UUID::generateUUID()); //Cassandra does NOT like to have a number at the beginning of a table name
        std::replace(id_object_str.begin(), id_object_str.end(), '-','_'); //Cassandra does NOT like character '-' in table names
    } else {
        id_object_str = std::string(id_object);
    }
    std::string name(config["execution_name"] + "." + id_object_str);
    uint64_t *c_uuid = UUID::generateUUID5(name.c_str()); // UUID for the new object

    switch(oType.getType()) {
        case ObjSpec::valid_types::STORAGEOBJ_TYPE:
            {
                // StorageObj case
                //  Create table 'class_name' "CREATE TABLE ksp.class_name (storage_id UUID, nom typ, ... PRIMARY KEY (storage_id))"
                std::string query = "CREATE TABLE IF NOT EXISTS " +
                    config["execution_name"] + "." + id_model +
                    oType.table_attr;

                CassError rc = run_query(query);
                if (rc != CASS_OK) {
                    if (rc == CASS_ERROR_SERVER_INVALID_QUERY) { // keyspace does not exist
                        std::cout<< "HecubaSession::createObject: Keyspace "<< config["execution_name"]<< " not found. Creating keyspace." << std::endl;
                        std::string create_keyspace = std::string(
                                "CREATE KEYSPACE IF NOT EXISTS ") + config["execution_name"] +
                            std::string(" WITH replication = ") +  config["replication"];
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
                std::string insquery = std::string("INSERT INTO ") +
                    std::string("hecuba.istorage") +
                    std::string("(storage_id, name, class_name, columns)") +
                    std::string("VALUES ") +
                    std::string("(") +
                    UUID::UUID2str(c_uuid) + std::string(", ") +
                    "'" + name + "'" + std::string(", ") +
                    "'" + FQid_model + "'" + std::string(", ") +
                    oType.getColsStr() +
                    std::string(")");
                run_query(insquery);

                //  Create Writer for storageobj
                std::vector<config_map>* keyNamesDict = oType.getKeysNamesDict();
                std::vector<config_map>* colNamesDict = oType.getColsNamesDict();

                CacheTable *dataAccess = storageInterface->make_cache(generateTableName(FQid_model).c_str(), config["execution_name"].c_str(),
                          *keyNamesDict, *colNamesDict,
                          config);
                delete keyNamesDict;
                delete colNamesDict;
                o = new IStorage(this, FQid_model, config["execution_name"] + "." + id_object_str, c_uuid, dataAccess);

            }
            break;
        case ObjSpec::valid_types::STORAGEDICT_TYPE:
            {
                // Dictionary case
                //  Create table 'name' "CREATE TABLE ksp.name (nom typ, nom typ, ... PRIMARY KEY (nom, nom))"
                bool new_element = true;
                std::string query = "CREATE TABLE " +
                    config["execution_name"] + "." + id_object_str +
                    oType.table_attr;

                CassError rc = run_query(query);
                if (rc != CASS_OK) {
                    if (rc == CASS_ERROR_SERVER_ALREADY_EXISTS ) {
                        new_element = false; //OOpps, creation failed. It is an already existent object.
                    } else if (rc == CASS_ERROR_SERVER_INVALID_QUERY) {
                        std::cout<< "HecubaSession::createObject: Keyspace "<< config["execution_name"]<< " not found. Creating keyspace." << std::endl;
                        std::string create_keyspace = std::string(
                                "CREATE KEYSPACE IF NOT EXISTS ") + config["execution_name"] +
                            std::string(" WITH replication = ") +  config["replication"];
                        rc = run_query(create_keyspace);
                        if (rc != CASS_OK) {
                            std::string msg = std::string("HecubaSession::createObject: Error creating keyspace ") + create_keyspace;
                            throw ModuleException(msg);
                        } else {
                            rc = run_query(query);
                            if (rc != CASS_OK) {
                                if (rc == CASS_ERROR_SERVER_ALREADY_EXISTS) {
                                    new_element = false; //OOpps, creation failed. It is an already existent object.
                                }  else {
                                    std::string msg = std::string("HecubaSession::createObject: Error executing query ") + query;
                                    throw ModuleException(msg);
                                }
                            }
                        }
                    } else {
                        std::string msg = std::string("HecubaSession::createObject: Error executing query ") + query;
                        throw ModuleException(msg);
                    }
                }

                if (new_element) {
                    //  Add entry to hecuba.istorage: TODO add the tokens attribute
                    // TODO EXPECTED:vvv NOW HARDCODED
                    //classname = FQid_model
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
                        UUID::UUID2str(c_uuid) + std::string(", ") +
                        "'" + name + "'" + std::string(", ") +
                        "'" + FQid_model + "'" + std::string(", ") +
                        oType.getKeysStr() + std::string(", ") +
                        oType.getColsStr() +
                        std::string(")");
                    run_query(insquery);
                } else {
                    std::cerr << "WARNING: Object "<<id_object_str<<" already exists. Trying to overwrite it. It may fail if the schema does not match."<<std::endl;
                    // TODO: THIS IS NOT WORKING. We need to get the storage_id (c_uuid) from istorage DISABLE
                    // TODO: Check the schema in Cassandra matches the model
                }

                //  Create Writer for dictionary
                std::vector<config_map>* keyNamesDict = oType.getKeysNamesDict();
                std::vector<config_map>* colNamesDict = oType.getColsNamesDict();


                std::string topic = std::string(UUID::UUID2str(c_uuid));

                CacheTable *reader = storageInterface->make_cache(id_object_str.c_str(), config["execution_name"].c_str(),
                        *keyNamesDict, *colNamesDict,
                        config);

                delete keyNamesDict;
                delete colNamesDict;
                o = new IStorage(this, FQid_model, config["execution_name"] + "." + id_object_str, c_uuid, reader);
                DBG("HecubaSession::createObject: CREATED NEW STORAGEDICT with uuid "<< topic);
                if (oType.isStream()) {
                    DBG("     AND IT IS AN STREAM!");
                    o->configureStream(topic);
                }
            }
            break;

        case ObjSpec::valid_types::STORAGENUMPY_TYPE:
            {
#if 0
                // Create table
                std::string query = "CREATE TABLE IF NOT EXISTS " + config["execution_name"] + "." + id_object_str +
                    " (storage_id uuid, cluster_id int, block_id int, payload blob, "
                    "PRIMARY KEY((storage_id,cluster_id),block_id)) "
                    "WITH compaction = {'class': 'SizeTieredCompactionStrategy', 'enabled': false};";

                this->run_query(query);

                // StorageNumpy
                ArrayDataStore *array_store = new ArrayDataStore(id_object_str.c_str(), config["execution_name"].c_str(),
                        this->storageInterface->get_session(), config);
                //std::cout << "DEBUG: HecubaSession::createObject After ArrayDataStore creation " <<std::endl;

                // Create entry in hecuba.istorage for the new numpy
                ArrayMetadata numpy_metas;
                getMetaData(metadata, numpy_metas); // numpy_metas = getMetaData(metadata);
                DBG("DEBUG: HecubaSession::createNumpy . Size "<< numpy_metas.get_array_size());
                //std::cout<< "DEBUG: HecubaSession::createObject After metadata creation " <<std::endl;

                registerNumpy(numpy_metas, name, c_uuid);

                //std::cout<< "DEBUG: HecubaSession::createObject After REGISTER numpy into ISTORAGE" <<std::endl;

                //Create keys, values to store the numpy
                //double* tmp = (double*)value;
                //std::cout<< "DEBUG: HecubaSession::createObject BEFORE store FIRST element in NUMPY is "<< *tmp << " and second is "<<*(tmp+1)<< " 3rd " << *(tmp+2)<< std::endl;
                array_store->store_numpy_into_cas(c_uuid, numpy_metas, value);

                o = new IStorage(this, FQid_model, config["execution_name"] + "." + id_object_str, c_uuid, array_store->getWriteCache());
                std::string topic = std::string(UUID::UUID2str(c_uuid));
                DBG("HecubaSession::createObject: CREATED NEW STORAGENUMPY with uuid "<< topic);
                o->setNumpyAttributes(array_store, numpy_metas,value);
                if (oType.isStream()) {
                    DBG("     AND IT IS AN STREAM!");
                    o->configureStream(topic);
                }

#else
		throw ModuleException("HECUBA SESSION: create object storagenumpy (use new interface)");
#endif
            }
            break;
        default:
            throw ModuleException("HECUBA Session: createObject Unknown type ");// + std::string(oType.objtype));
            break;
    }
    //std::cout << "DEBUG: HecubaSession::createObject DONE" << std::endl;
    return o;
}
#endif
//returns true if the class_name was not inserted and false otherwise
bool HecubaSession::registerClassName(const std::string& class_name) {
    std::pair<std::map<std::string,char>::iterator,bool> res = registeredClasses.insert(std::pair<std::string,char>(class_name,'c'));
    return res.second;
}
bool HecubaSession::registerObject(const std::shared_ptr<CacheTable> c, const std::string& class_name) {
    alive_objects.push_back(c);
    deallocateObjects(); // check if it is possible to deallocate some objects
    return registerClassName(class_name);
}
bool HecubaSession::registerObject(const std::shared_ptr<ArrayDataStore> a, const std::string& class_name) {
    alive_numpy_objects.push_back(a);
    deallocateObjects(); // check if it is possible to deallocate some objects
    return registerClassName(class_name);
}

void HecubaSession::deallocateObjects() {
    for (std::list<std::shared_ptr<CacheTable>>::iterator it = alive_objects.begin(); it != alive_objects.end();) {
        std::shared_ptr<CacheTable> t = *it;
        //std::cout << "LIST: "<< t.get() <<" ("<<t.use_count()<<")"<<std::endl;
        if (t.use_count() == 2) { // The object has been "destroyed" from its use: 2 references variable t and alive_objects
            if (t->get_writer()->is_write_completed()) { // Have the pending writes completed?
                //std::cout << "DELETE FROM LIST: "<< t.get() <<" ("<<t.use_count()<<")"<<std::endl;
                it = alive_objects.erase(it);
            } else {
                it++;
            }
        }else {
            it ++;
        }
    }
    for (std::list<std::shared_ptr<ArrayDataStore>>::iterator it = alive_numpy_objects.begin(); it != alive_numpy_objects.end();) {
        std::shared_ptr<ArrayDataStore> t = *it;
        //std::cout << "LIST: "<< t.get() <<" ("<<t.use_count()<<")"<<std::endl;
        if (t.use_count() == 2) { // The object has been "destroyed" from its use: 2 references variable t and alive_numpy_objects
            if (t->getWriteCache()->get_writer()->is_write_completed()) { // Have the pending writes completed?
                //std::cout << "DELETE FROM LIST: "<< t.get() <<" ("<<t.use_count()<<")"<<std::endl;
                it = alive_numpy_objects.erase(it);
            } else {
                it++;
            }
        }else {
            it ++;
        }
    }
}

std::string HecubaSession::getExecutionName() {
    return config["execution_name"];
}
