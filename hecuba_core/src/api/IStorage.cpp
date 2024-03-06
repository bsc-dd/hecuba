#include "IStorage.h"
#include "UUID.h"

#include <boost/uuid/uuid.hpp>
#include <typeinfo>
#include "HecubaExtrae.h"

#define ISKEY true


IStorage::IStorage() {
    HecubaExtrae_event(HECUBAEV, HECUBA_IS|HECUBA_INSTANTIATION);
    DBG( "default constructor this "<< this );
    try {
        // Start cassandra connection as soon as possible to speedup the global startup (removing the cassandra connection from the critical path)
	    HecubaSession& currentSession = getCurrentSession();
    } catch (ModuleException &e) {
        // Cassandra is not up yet... continuing with volatile objects...
    }
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}

IStorage::~IStorage() {
    HecubaExtrae_event(HECUBAEV, HECUBA_IS|HECUBA_DESTROY);
    DBG( UUID::UUID2str(getStorageID())<<" this: "<< this);
    if (storageid != nullptr) {
        free (storageid);
        storageid = nullptr;
    }
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}

IStorage::IStorage(const IStorage& src) {
    HecubaExtrae_event(HECUBAEV, HECUBA_IS|HECUBA_INSTANTIATION);
    DBG(" copy constructor this " << this << " src " << &src );
    *this = src;
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}

IStorage& IStorage::operator = (const IStorage& src) {
    HecubaExtrae_event(HECUBAEV, HECUBA_IS|HECUBA_ASSIGNMENT);
    DBG(" IStorage::copy operator = this "<< this << " src "<<&src);
    if (this != &src) {
        IStorageSpec = src.IStorageSpec;
        pythonSpec = src.pythonSpec;
        tableName = src.tableName;
        pending_to_persist = src.pending_to_persist;
        persistent = src.persistent;
        keysnames = src.keysnames;
        keystypes = src.keystypes;
        colsnames = src.colsnames;
        colstypes = src.colstypes;

        if (this->storageid != nullptr){
            free(this->storageid);
        }
        if (src.storageid != nullptr) {
            this->storageid = (uint64_t *)malloc(2*sizeof(uint64_t));
            memcpy(this->storageid, src.storageid, 2* sizeof(uint64_t));
        } else {
            this->storageid = nullptr;
        }

        id_obj = src.id_obj;
        id_model = src.id_model;
        class_name = src.class_name;

        streamEnabled = src.streamEnabled;

        dataWriter = src.dataWriter;
        dataAccess = src.dataAccess;

        //partitionKeys = src.partitionKeys;
        //clusteringKeys = src.clusteringKeys;
        //valuesDesc = src.valuesDesc;
        delayedObjSpec = src.delayedObjSpec;
    }
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
    return *this;
}

std::string IStorage::generate_numpy_table_name(std::string attributename) {
    /* ksp.DUUIDtableAttribute extracted from hdict::make_val_persistent */
    //std::cout << "DEBUG: IStorage::generate_numpy_table_name: BEGIN attribute:"<<attributename<<std::endl;
    std::string name;
    // Obtain keyspace and table_name from id_obj (keyspace.table_name)
    uint32_t pos = id_obj.find_first_of(".");
    //std::string ksp = id_obj.substr(0, pos);
    std::string table_name = id_obj.substr(pos+1, id_obj.length()); //skip the '.'

    // Generate a new UUID for this attribute
    uint64_t *c_uuid = UUID::generateUUID(); // UUID for the new object
    std::string uuid = UUID::UUID2str(c_uuid);
    for(auto i=0; i!=uuid.size(); i++) {
        if (uuid[i] == '-') uuid[i] = '_';
    }

    // attributename contains "name1.name2.....attributename" therefore keep the last attribute name TODO: Fix this 'name.name...' nightmare
    //name = ksp + ".D" + uuid + table_name + attributename.substr(attributename.find_last_of('_'), attributename.size());
    name = "D" + uuid + table_name + attributename;

    name = name.substr(0,48);
    //std::cout << "DEBUG: IStorage::generate_numpy_table_name: END "<<name<<std::endl;
    return name;
}

uint64_t* IStorage::getStorageID() {
    return storageid;
}

const std::string& IStorage::getName() const {
    return id_obj;
}

const std::string& IStorage::getTableName() const {
    return tableName;
}
void
IStorage::sync(void) {
    HecubaExtrae_event(HECUBAEV, HECUBA_IS|HECUBA_SYNC);
    this->getDataWriter()->flush_elements();
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}


/* Transform IStorage pointers to StorageIDs pointers
    Args:
        dst: block of memory to update
        value_type: string from ObjSpec specifying the type to transform,
            anything different from a basic type will be transformed
        src: pointer to a block of memory with a value (if basic type), a
            pointer to char (string) or a pointer to an IStorage object
        src_size: size of src


     src ----> +---------+
               |     *---+------------------->+----------+
               +---------+                    | IStorage-+-------->+-----------+
                                              +----------+         | StorageID |
    Generates:                                                     +-----------+

     dst ----> +---------+
               |     *---+------------------->+-----------+ (This memory is allocated and contains a copy)
               +---------+                    | StorageID'|
                                              +-----------+
    But:
     src ----> +---------+
               | 42      |
               +---------+
    Generates:

     dst ----> +---------+
               | 42      |
               +---------+

*/

bool IStorage::convert_IStorage_to_UUID(char * dst, const std::string& value_type, const void* src, int64_t src_size) const {
    bool isIStorage = false;
    void * result;
    DBG( "IStorage::convert_IStorage_to_UUID " + value_type );
    if (!ObjSpec::isBasicType(value_type)) {
        DBG( "IStorage::convert_IStorage_to_UUID NOT BASIC" );
        result = (*(IStorage **)src)->getStorageID(); // 'src' MUST be a valid pointer or it will segfault here...
#if 0
        // Minimal Check for UUID
        boost::uuids::uuid u;
        memcpy(&u, result, 16);
        boost::uuids::uuid::variant_type variant = u.variant();
        boost::uuids::uuid::version_type version = u.version();
        if ( ! ((variant == boost::uuids::uuid::variant_rfc_4122) && (version == boost::uuids::uuid::version_name_based_sha1))) {
            throw ModuleException("IStorage:: Set Item. Wrong UUID format for object... is it a pointer to an IStorage?");
        }
#endif
        // It seems like a valid UUID
        void * sid = malloc(sizeof(uint64_t)*2);
        memcpy(sid, result, sizeof(uint64_t)*2);
        memcpy(dst, &sid, src_size) ;
        isIStorage = true;
    } else{ // it is a basic type, just copy the value
        //if value_type is a string, copy it to a new variable to be independent of potential memory free from the user code
        DBG( "IStorage::convert_IStorage_to_UUID BASIC is " + value_type );
        if (!value_type.compare(std::string{"text"})) {
            result = *(char**)src; // 'src' MUST be a valid pointer or it will segfault here...
            char* tmp = (char*)malloc (strlen((char*)result)+1);
            memcpy(tmp, (char*)result, strlen((char*)result)+1); // Copy the string content
            memcpy(dst, &tmp, src_size); // Copy the address
        } else {
            memcpy(dst, src, src_size); // Copy the content
        }
    }
    return isIStorage;
}

/* deep_copy_attribute_buffer: Creates a copy of a block of memory containing values to store in a table, complex types are also copied.
 *  iskey   : The buffer corresponds to a key? or is it a column?
 *  src     : pointer to source memory block
 *  src_size: length of the source memory block
 *  num_attrs: Number of attributes inside the 'src' memory block
 * return a NEW block of memory with the same content as 'src' but creating NEW copies for internal complex data (currently STRINGS).
 */
void * IStorage::deep_copy_attribute_buffer(bool isKey, const void* src, uint64_t src_size, uint32_t num_attrs) {

    /** WARNING: The 'src' buffer comes from user, therefore the fields order is
     * specified by the ObjSpec which may or may not (possibly the latter)
     * coincide with the format needed to access the database.
     * This method reorders the resulting buffer to be suitable for this
     * access.*/

    void * dst = malloc(src_size);

    // Process src to generate memory to complex types: UUIDs, strings,...
    const ObjSpec ospec = getObjSpec();

    const TableMetadata* writerMD = dataWriter->get_metadata();

    DBG( "IStorage::deep_copy_attribute_buffer num attributes="<<num_attrs);
    int64_t value_size;
    uint64_t offset=0;

    // Traverse the buffer following the user order...
    for (uint32_t i=0; i < num_attrs; i++) {

        DBG("  IStorage::deep_copy_attribute_buffer offset ="<<offset);

        std::string column_name;
        std::string value_type;
        const ColumnMeta *c;
        // Only 2 cases supported: keys or values
        if (isKey) {
            column_name = ospec.getIDObjFromKey(i);
            value_type = ospec.getIDModelFromKey(i);
            c = writerMD->get_single_key(column_name);
        } else {
            column_name = ospec.getIDObjFromCol(i);
            value_type = ospec.getIDModelFromCol(i);
            c = writerMD->get_single_column(column_name);
        }
        value_size = c->size;

        // Convert each attribute and REORDER it to the right position in cassandra tables...
        convert_IStorage_to_UUID(((char *)dst)+c->position, value_type, ((char*)src) + offset, value_size);

        offset += value_size;
    }

    return dst;
}



void
IStorage::send_values(const void* value) {
	// This will be redefined by underlying subclasses
    DBG("IStorage::send_values");
}
// this comment beloow is valid for both setItem (StorageDict.h) and setAttr (StorageObject.h). Find a proper place for it.
/* Args:
    key and value are pointers to a block of memory with the values (if basic types) or pointers to IStorage or strings:
    key/value -.
               |
               V
               +---------+
               |     *---+------------------->+----------+
               +---------+                    | IStorage |-------->+-----------+
               | 42      |                    +----------+         | StorageID |
               +---------+                                         +-----------+
               | 0.66    |
               +---------+
               |     *---+------------------->+--------+
               +---------+                    | hola\0 |
                                              +--------+

    Create a copy of it, normalizing the pointer to other structs keeping just the storageID:
    cc_key/cc_val
               |
               V
               +---------+
               |     *---+------------------->+-----------+ (newly allocated and copied)
               +---------+                    | StorageID'|
               | 42      |                    +-----------+
               +---------+
               | 0.66    |
               +---------+
               |     *---+------------------->+--------+ (newly allocated and copied)
               +---------+                    | hola\0'|
                                              +--------+

    Therefore, 'key' and 'value' may be freed after this method.
*/

void IStorage::extractFromQueryResult(std::string value_type, uint32_t value_size, void *query_result, void *valuetoreturn) const{
    if (!ObjSpec::isBasicType(value_type)) {
        uint64_t *uuid = *(uint64_t **) query_result;
        char *tmp = (char *) malloc(sizeof(uint64_t)*2);
        memcpy(tmp, uuid, sizeof(uint64_t)*2);
        memcpy(valuetoreturn, &tmp, sizeof(uint64_t*));
    } else {
        if (value_type == "text") {
            char *str = *(char**)query_result;
            value_size = strlen(str) + 1;
            char *tmp = (char *) malloc(value_size);
            memcpy(tmp, str, value_size);
            memcpy(valuetoreturn, &tmp, sizeof(char*));
        } else {
            memcpy(valuetoreturn, query_result, value_size);
        }
    }
}

/* Given a result from a cassandra query, extract all elements into valuetoreturn buffer*/
/* type = KEYS/COLUMNS TODO: add ALL to support the iteration for both keys and values (pythom items method) */

void IStorage::extractMultiValuesFromQueryResult(void *query_result, void *valuetoreturn, int type) {
    uint32_t attr_size;
    ObjSpec ospec = getObjSpec();

    const TableMetadata* writerMD = dataWriter->get_metadata();

    uint64_t offset = 0; // offset in user buffer
    std::shared_ptr<const std::vector<ColumnMeta> > metas;
    if (type == COLUMNS) {
        metas = writerMD->get_values();
        attr_size = writerMD->get_values_size();
    } else {
        metas = writerMD->get_keys();
        std::pair<uint16_t,uint16_t> keys_size = writerMD->get_keys_size();
        attr_size=keys_size.first + keys_size.second;
    }

    char *valuetmp = (char*) malloc(attr_size);

    std::string attr_name;
    std::string attr_type;
    const ColumnMeta *c;
    for (uint64_t pos = 0; pos<metas->size(); pos++) {
        if (type == COLUMNS) {
            attr_name = ospec.getIDObjFromCol(pos);
            attr_type = ospec.getIDModelFromCol(pos);
            c = writerMD->get_single_column(attr_name);
        } else {
            attr_name = ospec.getIDObjFromKey(pos);
            attr_type = ospec.getIDModelFromKey(pos);
            c = writerMD->get_single_key(attr_name);
        }
        attr_size = c->size;

        // c->position contains offset in query_result.
        extractFromQueryResult(attr_type, attr_size, ((char*)query_result) + c->position, valuetmp+offset);

        offset += attr_size;
    }

    // Copy Result to user:
    //   If a single basic type value is returned then the user passes address
    //   to store the value, otherwise we allocate the memory to store all the
    //   values.
    if (metas->size() == 1) {
        memcpy(valuetoreturn, valuetmp, attr_size);
    } else {
        memcpy(valuetoreturn, &valuetmp, sizeof(char*));
    }
}

void IStorage::enableStream() {
    streamEnabled=true;
}

bool IStorage::isStream() {
    return streamEnabled;
}

void IStorage::configureStream(std::string topic) {
	enableStream();
    this->getDataWriter()->enable_stream(topic.c_str(),(std::map<std::string, std::string>&)getCurrentSession().config);
}


void IStorage::setClassName(std::string name) {
	this->class_name = name;
}
const std::string& IStorage::getClassName() {
	return this->class_name;
}
const std::string& IStorage::getIdModel() {
	return this->id_model;
}

/* split FQIDmodel ("keyspace.tablename") into its 2 components: keyspace and tablename */
std::pair<std::string, std::string> IStorage:: getKeyspaceAndTablename( const std::string& FQIDmodel ) const {
	std::string keyspace = FQIDmodel;
	std::string tablename;

	uint32_t pos = keyspace.find_first_of('.');
	tablename = keyspace.substr(pos+1);
	keyspace = keyspace.substr(0,pos);
	return std::make_pair(keyspace, tablename);
}

void IStorage::setIdModel(std::string name) {
	this->id_model=name;
}

HecubaSession&  IStorage::getCurrentSession() const{
	return(HecubaSession::get());
}

// file name: execution_name_class_name+.py
// current directory
// if the file already exists appends the new definition at the end

void IStorage::writePythonSpec() {
        HecubaExtrae_event(HECUBADBG, HECUBA_WRITEPYTHONSPEC);
        std::string pythonFileName =  this->getClassName() + ".py";
        std::ofstream fd(pythonFileName);
        fd << this->getPythonSpec();
        fd.close();
        HecubaExtrae_event(HECUBADBG, HECUBA_END);

}
void IStorage::setObjSpec(const ObjSpec &oSpec) {this->IStorageSpec=oSpec;}

ObjSpec& IStorage::getObjSpec() {
    if (delayedObjSpec) { // TODO Check that this is not happening and remove the delayedObjSpec from IStorage
        throw ModuleException("OOOOppps IStorage::getObjSpec called with delayed ObjSpec. This should not happen");
    }
    return IStorageSpec;
}


void IStorage::setPythonSpec(std::string pSpec) {this->pythonSpec=pSpec;}
std::string IStorage::getPythonSpec() {
	if (this->pythonSpec.empty()) {
		this->generatePythonSpec();
	}
	return this->pythonSpec;

}
void IStorage::setObjectName(std::string id_obj) {
	this->id_obj = id_obj;
}

std::string IStorage::getObjectName() {
	return (this->id_obj);
}
std::string IStorage::getTableName(){
	return(this->tableName);
}
void IStorage::setTableName(std::string tableName) {
	this->tableName = tableName;
}

bool IStorage::is_pending_to_persist() {
	return this->pending_to_persist;
}
void IStorage::set_pending_to_persist() {
    this->pending_to_persist=true;
}

void IStorage::make_persistent(const std::string  id_obj) {
    HecubaExtrae_event(HECUBAEV, HECUBA_IS|HECUBA_MK_PERSISTENT);

	std::string id_object_str;
	//if the object is not registered, i.e. the class_name is empty, we return error

	if (class_name.empty()) {
		throw std::runtime_error("Trying to persist a non-registered object with name "+ id_obj);

	}


    HecubaSession& currentSession = getCurrentSession();
	if (id_obj.empty()) { //No name used
		throw std::runtime_error("Trying to persist with an empty name ");
	} else {
		id_object_str = std::string(id_obj);
        size_t pos= id_object_str.find_first_of(".");
        if (pos == std::string::npos) {// Not found
            id_object_str = currentSession.config["execution_name"] + "." + id_object_str;
        } else {
            pos = id_object_str.find_first_of(".", pos);
            if (pos != std::string::npos) {// Found
		        throw std::runtime_error("Trying to persist with a name with multiple 'dots' ["+ id_object_str +"]");
            }
        }
	}

	uint64_t *c_uuid=UUID::generateUUID5(id_object_str.c_str()); // UUID for the new object

	init_persistent_attributes(id_object_str, c_uuid) ;

	ObjSpec oType = this->getObjSpec();

	bool new_element = true;
    if (is_create_table_required()){ // If we are a Numpy sharing the table, we do NOT need to create the table (as it is shared and already created)
        std::string query = "CREATE TABLE " +
            currentSession.config["execution_name"] + "." + this->tableName +
            oType.table_attr;

        HecubaExtrae_event(HECUBACASS, HBCASS_CREATE);
        CassError rc = currentSession.run_query(query);
        if (rc != CASS_OK) {
            if (rc == CASS_ERROR_SERVER_ALREADY_EXISTS ) {
                new_element = false; //OOpps, creation failed. It is an already existent object.
            } else if (rc == CASS_ERROR_SERVER_INVALID_QUERY) {
                std::cerr<< "IStorage::make_persistent: Keyspace "<< currentSession.config["execution_name"]<< " not found. Creating keyspace." << std::endl;
                std::string create_keyspace = std::string(
                        "CREATE KEYSPACE IF NOT EXISTS ") + currentSession.config["execution_name"] +
                    std::string(" WITH replication = ") +  currentSession.config["replication"];
                rc = currentSession.run_query(create_keyspace);
                if (rc != CASS_OK) {
                    std::string msg = std::string("IStorage::make_persistent: Error creating keyspace ") + create_keyspace;
                    throw ModuleException(msg);
                } else {
                    rc = currentSession.run_query(query);
                    if (rc != CASS_OK) {
                        if (rc == CASS_ERROR_SERVER_ALREADY_EXISTS) {
                            new_element = false; //OOpps, creation failed. It is an already existent object.
                        }  else {
                            std::string msg = std::string("IStorage::make_persistent: Error executing query ") + query;
                            throw ModuleException(msg);
                        }
                    }
                }
            } else {
                std::string msg = std::string("IStorage::make_persistent: Error executing query ") + query;
                throw ModuleException(msg);
            }
        }
        HecubaExtrae_event(HECUBACASS, HBCASS_END);
    }

	// Create READ/WRITE cache accesses
	initialize_dataAcces();
    if (isStream()){
		getObjSpec().enableStream();
		configureStream(UUID::UUID2str(c_uuid));
	}

    HecubaExtrae_event(HECUBADBG, HECUBA_PERSIST_METADATA);
	persist_metadata(c_uuid); //depends on the type of persistent object
				// TODO: deal with the case of already existing dictionaries and manage new_element == false
    HecubaExtrae_event(HECUBADBG, HBCASS_END);


	// Now that the writer is created, persist data
    HecubaExtrae_event(HECUBADBG, HECUBA_PERSIST_DATA);
	persist_data();
    HecubaExtrae_event(HECUBADBG, HBCASS_END);

    DBG(" IStorage::make_persistent Object "<< id_model <<" with name "<< id_obj);
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}

/* id_object_str: User name for the object passed to make_persistent (includes keyspace)
   c_uuid       : UUID to use for this object */
void IStorage::init_persistent_attributes(const std::string& id_object_str, uint64_t* c_uuid) {
	this->setObjectName(id_object_str);
	this->assignTableName(id_object_str, getClassName()); //depends on the type of persistent object

	this->storageid = c_uuid;
	this->pending_to_persist=false;
	this->persistent=true;
}


/* Get name, classname and numpymetas from hecuba.istorage */
const struct IStorage::metadata_info IStorage::getMetaData(uint64_t* uuid) const {

	struct metadata_info res;

	void * localuuid = malloc(2*sizeof(uint64_t));
	memcpy(localuuid, uuid, 2*sizeof(uint64_t));
	void * key = malloc(sizeof(char*));
	memcpy(key, &localuuid, sizeof(uint64_t*));

    HecubaSession& currentSession = getCurrentSession();
	std::vector <const TupleRow*> result = currentSession.getHecubaIstorageAccess()->retrieve_from_cassandra(key);

	if (result.empty()) throw ModuleException("IStorage uuid "+UUID::UUID2str(uuid)+" not found. Unable to get its metadata.");

	uint32_t pos = currentSession.getHecubaIstorageAccess()->get_metadata()->get_columnname_position("name");
	char *keytable = *(char**)result[0]->get_element(pos); //Value retrieved from cassandra has 'keyspace.tablename' format


	// Check that retrieved classname form hecuba coincides with 'id_model'
	pos = currentSession.getHecubaIstorageAccess()->get_metadata()->get_columnname_position("class_name");
	char *classname = *(char**)result[0]->get_element(pos); //Value retrieved from cassandra has 'keyspace.tablename' format

	// Read the UDT case (numpy_meta)from the row retrieved from cassandra
	pos = currentSession.getHecubaIstorageAccess()->get_metadata()->get_columnname_position("numpy_meta");
	if (result[0]->get_element(pos) != 0) {
		ArrayMetadata *numpy_metas = *(ArrayMetadata**)result[0]->get_element(pos);
		res.numpy_metas = *numpy_metas;
		DBG("DEBUG: HecubaSession::createNumpy . Size "<< numpy_metas->get_array_size());
	}

	res.name = std::string( keytable );
	res.class_name = std::string( classname );
	return  res;
}

std::shared_ptr<CacheTable> IStorage::getDataAccess() const {
	return dataAccess;
}

void IStorage::setCache(CacheTable* cache) {
	dataAccess = std::shared_ptr<CacheTable>(cache);
	dataWriter = dataAccess->get_writer();
}

void IStorage::getByAlias(const std::string& name) {
    HecubaExtrae_event(HECUBAEV, HECUBA_IS|HECUBA_GET_BY_ALIAS);
	std::string FQname (getCurrentSession().config["execution_name"] + "." + name);
	uint64_t *c_uuid=UUID::generateUUID5(FQname.c_str()); // UUID for the new object

	setPersistence(c_uuid);
    if (isStream()) {
		getObjSpec().enableStream();
		configureStream(std::string(UUID::UUID2str(c_uuid)));
	}
    DBG(" IStorage::getByAlias object with name ["<<name<<"] and uuid ["<<UUID::UUID2str(c_uuid)<<"]");
    HecubaExtrae_event(HECUBAEV, HECUBA_END);
}

void IStorage::get_by_alias(const std::string& name) {
    getByAlias(name);
}

void IStorage::initializeClassName(std::string class_name) {
	std::string FQname;
	bool new_element = true;
	if (class_name == "StorageNumpy") {
		class_name="hecuba.hnumpy.StorageNumpy";
		FQname=class_name;
		new_element=false;
	} else {
		FQname = class_name + "." + class_name;
	}
	setIdModel(FQname);
	setClassName(class_name);
    DBG(" IStorage::initializeClassName [" << FQname<<"] with name ["<<getName()<<"]");
}
