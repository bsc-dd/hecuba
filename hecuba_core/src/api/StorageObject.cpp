#include "SO_Attribute.h"
#include "Hecuba_StorageObject.h"
#include "debug.h"
#include "HecubaExtrae.h"


// Example of definition of user class that implements a StorageObject
// class mySOClass: public StorageObject {
//
//      HECUBA_ATTRS(
//              int, a,
//              float, b,
//              std::string, c
//              )
//
// };
// the HECUBA_ATTRS macro defines the attributes a,b and c, with the initialization with the StorageObject and the name of the attributes
// and the name of the class
// class mySOClass: public StorageObject {
//      SO_ClassName myname  ={this, typeid(*this).name()};  --> sets the class name of the StorageObject
//      SO_Attribute<int> a = {this, "a"};  --> this, to get access to the storageObject and enable the interaction with Cassandra
//                                          --> the string with the name of the attribute to be used in the setAttr an getAttr methods
//      SO_Attribute<float> b = {this, "b"};
//      SO_Attribute<std::string> c = {this, "c"};
//
//}


StorageObject::StorageObject(): IStorage(){
    HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_INSTANTIATION);
    DBG(" constructor without parameters " << this);
    delayedObjSpec = true;
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}


// c++ only calls implicitly the constructor without parameters. To invoke this constructor we need to add to the user class an explicit call to this
StorageObject::StorageObject(const std::string& name): IStorage() {
    HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_INSTANTIATION);
    delayedObjSpec = true;
    setObjectName(name);
    set_pending_to_persist();
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}

StorageObject::StorageObject(const StorageObject& src): IStorage() {
    HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_INSTANTIATION);
    *this=src;
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}

StorageObject& StorageObject::operator = (const StorageObject& src) {
    HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_ASSIGNMENT);
    if (this != &src ){
        this->IStorage::operator=(src); //Inherit IStorage attributes
        //this->st = src.st;
        this->valuesDesc = src.valuesDesc;
        this->translate = src.translate;
    }
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
    return *this;
}

StorageObject::~StorageObject() {
    HecubaExtrae_event(HECUBAEV, HECUBA_SO|HECUBA_DESTROY);
    DBG( " "<<this);
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}


// It generates the python specification for the class during the registration of the object
void StorageObject::generatePythonSpec() {
    std::string StreamPart="";
    if (isStream() ){
        StreamPart=std::string(", StorageStream");
    }
    std::string pythonSpec = PythonDisclaimerString + "from hecuba import StorageObject"
        + StreamPart +
        + "\n\nclass "
        + getClassName() + "(StorageObject"
        + StreamPart
        + "):\n"
        + "   '''\n";

    std::string itemSpec = "";

    for (std::vector<std::pair<std::string,std::string>>::iterator it=valuesDesc.begin(); it!=valuesDesc.end(); ++it)
        itemSpec+="   @Classfield "+it->first + " " + ObjSpec::cass_to_hecuba(it->second) + "\n";

    pythonSpec += itemSpec + "   '''\n";

    setPythonSpec(pythonSpec);
}
void StorageObject::addAttrSpec(const std::string& type, const std::string& name) {
    std::pair<std::string, std::string> d = {name, ObjSpec::c_to_cass(type)};
    this->valuesDesc.push_back(d);
}

ObjSpec StorageObject::generateObjSpec() {
    ObjSpec soSpec;
    std::vector<std::pair<std::string, std::string>> partitionKeys; //empty
    std::vector<std::pair<std::string, std::string>> clusteringKeys; //empty
    partitionKeys.push_back(std::pair<std::string,std::string>("storage_id","uuid"));
    soSpec = ObjSpec(ObjSpec::valid_types::STORAGEOBJ_TYPE, partitionKeys, clusteringKeys, valuesDesc, "");
    return soSpec;
}

void StorageObject::assignTableName(const std::string& id_obj, const std::string& class_name) {
    this->setTableName(class_name);
}

void StorageObject::persist_metadata(uint64_t* c_uuid) {
    ObjSpec oType = getObjSpec();
    std::string insquery = 	std::string("INSERT INTO ") +
        std::string("hecuba.istorage") +
        std::string("(storage_id, name, class_name, columns)") +
        std::string("VALUES ") +
        std::string("(") +
        UUID::UUID2str(c_uuid) + std::string(", ") +
        "'" + getObjectName () + "'" + std::string(", ") +
        "'" + this->getIdModel() + "'" + std::string(", ") +
        oType.getColsStr() +
        std::string(")");
    HecubaExtrae_event(HECUBACASS, HBCASS_SYNCWRITE);
    CassError rc = getCurrentSession().run_query(insquery);
    HecubaExtrae_event(HECUBACASS, HBCASS_END);
    if (rc != CASS_OK) {
        std::string msg = std::string("StorageDict::persist_metadata: Error executing query ") + insquery;
        throw ModuleException(msg);
    }
}

/* setPersistence - Inicializes current instance to conform to uuid object. To be used on an empty instance. */
void StorageObject::setPersistence (uint64_t *uuid) {
    // FQid_model: Fully Qualified name for the id_model: module_name.id_model
    std::string FQid_model = this->getIdModel();    //"myclass.myclass"

    struct metadata_info row = this->getMetaData(uuid);

    std::pair<std::string, std::string> idObj = getKeyspaceAndTablename( row.name );  //"keysp.name'
    std::string keyspace = idObj.first;
    std::string tablename = idObj.first + "." + this->getClassName(); //"keysp.myclass"

    // Check that retrieved classname form hecuba coincides with 'id_model'
    std::string sobj_table_name = row.class_name;   //"myclass.myclass"

    // The class_name retrieved in the case of the storageobj is
    // the fully qualified name, but in cassandra the instances are
    // stored in a table with the name of the last part(example:
    // "model_complex.info" will have instances in "keyspace.info")
    // meaning that in a complex scenario with different models...
    // we will loose information. FIXME
    if (sobj_table_name.compare(FQid_model) != 0) {
        //throw ModuleException("HecubaSession::createObject uuid "+UUID::UUID2str(uuid)+" "+ tablename + " has unexpected class_name " + sobj_table_name + " instead of "+FQid_model);
    }

    init_persistent_attributes(tablename, uuid);
    // Create READ/WRITE cache accesses
    initialize_dataAcces();

}

void StorageObject::initialize_dataAcces() {
    //  Create Writer
    ObjSpec oType = this->getObjSpec();
    std::vector<config_map>* keyNamesDict = oType.getKeysNamesDict();
    std::vector<config_map>* colNamesDict = oType.getColsNamesDict();
    CacheTable *reader = getCurrentSession().getStorageInterface()->make_cache(this->getTableName().c_str(),
            getCurrentSession().config["execution_name"].c_str(), *keyNamesDict, *colNamesDict, getCurrentSession().config);
    this->setCache(reader);


    delete keyNamesDict;
    delete colNamesDict;
    if (getCurrentSession().registerObject(getDataAccess(),getClassName())) {
        writePythonSpec();
    }
}

/* Return:
 *  memory reference to datatype (must be freed by user) */
void StorageObject::getAttr(const std::string&  attr_name, void* valuetoreturn) {

    char *keytosend = (char*) malloc(sizeof(char*));
    char *uuidmem = (char*) malloc(sizeof(uint64_t)*2);
    const TableMetadata* writerMD = getDataWriter()->get_metadata();
    int value_size = writerMD->get_values_size(writerMD->get_columnname_position(attr_name));

    memcpy(keytosend, &uuidmem, sizeof(char*));
    memcpy(uuidmem, getStorageID(), sizeof(uint64_t)*2);

    std::vector<const TupleRow *> result = getDataAccess()->retrieve_from_cassandra(keytosend, attr_name.c_str());

    if (result.empty()) throw ModuleException("IStorage::getAttr: attribute " + attr_name + " not found in object " + getObjectName() );
    char *query_result= (char*)result[0]->get_payload();

    ObjSpec ospec = getObjSpec();
    std::string value_type = ospec.getIDModelFromColName(attr_name);

    extractFromQueryResult(value_type, value_size, query_result, valuetoreturn);

    // Free the TupleRows...
    for(auto i:result) {
        delete(i);
    }
    return;
}

void StorageObject::setAttr(const std::string& attr_name, void* value) {
    /* PRE: value arrives already coded as expected: block of memory with pointers to IStorages or basic values*/
    //std::cout << "DEBUG: IStorage::setAttr: "<<std::endl;
    //writeTable(attr_name, value, SETATTR_TYPE);

    void * cc_val;
    const TableMetadata* writerMD = getDataWriter()->get_metadata();
    DBG( "StorageObject::setAttr enter" );
    ObjSpec ospec = getObjSpec();
    int64_t value_size = writerMD->get_single_column(attr_name)->size;
    cc_val = malloc(value_size); // This memory will be freed after the execution of the query (at callback)
    std::string value_type = ospec.getIDModelFromColName(std::string(attr_name));
    convert_IStorage_to_UUID((char *)cc_val, value_type, value, value_size);
    uint64_t* sid = this->getStorageID();

    void* c_key = malloc(2*sizeof(uint64_t)); //uuid
    std::memcpy(c_key, sid, 2*sizeof(uint64_t));

    void *cc_key = malloc(sizeof(uint64_t *)); // This memory will be freed after the execution of the query (at callback)
    std::memcpy(cc_key, &c_key, sizeof(uint64_t *));

    getDataWriter()->write_to_cassandra(cc_key, cc_val, attr_name.c_str());
}

void StorageObject::setAttr(const std::string& attr_name, IStorage* value) {
    /* 'setAttr' expects a block of memory with pointers to IStorages, therefore add an indirection */
    setAttr(attr_name, (void *) &value);
}

ObjSpec& StorageObject::getObjSpec() {
    if (delayedObjSpec) {
        //only StorageObjects can have a delayedObjSpec because during the constructor maybe the attributes specification is unknown
        setObjSpec(generateObjSpec());
        initializeClassName (getClassName());
        delayedObjSpec = false;
    }
    return IStorage::getObjSpec();
}

