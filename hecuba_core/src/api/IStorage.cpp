#include "IStorage.h"
#include "UUID.h"

#include <regex>
#include <boost/uuid/uuid.hpp>

IStorage::IStorage(HecubaSession* session, std::string id_model, std::string id_object, uint64_t* storage_id, Writer* writer) {
	this->currentSession = session;
	this->id_model = id_model;
	this->id_obj = id_object;

	this->storageid = storage_id;
	this->dataWriter = writer;
}

IStorage::~IStorage() {
	delete(dataWriter);
}


std::string IStorage::generate_numpy_table_name(std::string attributename) {
    /* ksp.DUUIDtableAttribute extracted from hdict::make_val_persistent */
    //std::cout << "DEBUG: IStorage::generate_numpy_table_name: BEGIN attribute:"<<attributename<<std::endl;
    std::regex what("-");
    std::string name;
    // Obtain keyspace and table_name from id_obj (keyspace.table_name)
    uint32_t pos = id_obj.find_first_of(".");
    //std::string ksp = id_obj.substr(0, pos);
    std::string table_name = id_obj.substr(pos+1, id_obj.length()); //skip the '.'

    // Generate a new UUID for this attribute
    uint64_t *c_uuid = UUID::generateUUID(); // UUID for the new object
    std::string uuid = std::regex_replace(UUID::UUID2str(c_uuid), what, "_");

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

Writer *
IStorage::getDataWriter(void) {
    return dataWriter;
}

void
IStorage::sync(void) {
    this->dataWriter->flush_elements();
}


/* Transform IStorage pointers to StorageIDs pointers
    Args:
        dst: block of memory to update
        value_type: string from ObjSpec specifying the type to transform,
            anything different from a basic type will be transformed
        src: block of memory to transform
        src_size: size of src

    value is a pointer to a block of memory with a value (if basic types) or pointer to the IStorage:
     src ----> +---------+
               |     *---+------------------->+----------+
               +---------+                    | IStorage-+-------->+-----------+
                                              +----------+         | StorageID |
    Generates:                                                     +-----------+

     dst ----> +---------+
               |     *---+------------------->+-----------+
               +---------+                    | StorageID |
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
void IStorage::convert_IStorage_to_UUID(char * dst, const std::string& value_type, void* src, int64_t src_size) const {
    void * result;
    if (!ObjSpec::isBasicType(value_type)) {
        result = (*(IStorage **)src)->getStorageID(); // 'src' MUST be a valid pointer or it will segfault here...
        // Minimal Check for UUID
        boost::uuids::uuid u;
        memcpy(&u, result, 16);
        boost::uuids::uuid::variant_type variant = u.variant();
        boost::uuids::uuid::version_type version = u.version();
        if ( ! ((variant == boost::uuids::uuid::variant_rfc_4122) && (version == boost::uuids::uuid::version_name_based_sha1))) {
            throw ModuleException("IStorage:: Set Item. Wrong UUID format for object... is it a pointer to an IStorage?");
        }
        // It seems like a valid UUID
        void * sid = malloc(sizeof(uint64_t)*2);
        memcpy(sid, result, sizeof(uint64_t)*2);
        memcpy(dst, &sid, src_size) ;
    } else{ // it is a basic type, just copy the value
        memcpy(dst, src, src_size) ;
    }
}

/* Args:
    key and value are pointers to a block of memory with the values (if basic types) or pointer to the IStorage:
    key/value -+
               |
               V
               +---------+
               |     *---+------------------->+----------+
               +---------+                    | IStorage |-------->+-----------+
               | 42      |                    +----------+         | StorageID |
               +---------+                                         +-----------+
               | 0.66    |
               +---------+
               |     *---+------------------->+----------+
               +---------+                    | IStorage |-------->+-----------+
                                              +----------+         | StorageID |
                                                                   +-----------+
    Create a copy of it, normalizing the pointer to other structs keeping just the storageID:
    cc_key/cc_val
               |
               V
               +---------+
               |     *---+------------------->+-----------+
               +---------+                    | StorageID |
               | 42      |                    +-----------+
               +---------+
               | 0.66    |
               +---------+
               |     *---+------------------->+-----------+
               +---------+                    | StorageID |
                                              +-----------+
*/
void
IStorage::writeTable(const void* key, void* value, const enum IStorage::valid_writes mytype) {
	/* PRE: key and value arrives already coded as expected */

    void * cc_val;

    const TableMetadata* writerMD = dataWriter->get_metadata();

    DataModel* model = this->currentSession->getDataModel();

    ObjSpec ospec = model->getObjSpec(this->id_model);
    //std::cout << "DEBUG: IStorage::setItem: obtained model for "<<id_model<<std::endl;

    std::string value_type;
    if (ospec.getType() == ObjSpec::valid_types::STORAGEDICT_TYPE) {
        if (mytype != SETITEM_TYPE) {
            throw ModuleException("IStorage:: Set Item on a non Dictionary is not supported");
        }
        // Dictionary values may have N  columns, create a new structure with all of them normalized.
        std::cout<< "WriteTable malloc("<<writerMD->get_values_size()<<")"<<std::endl;
        cc_val = malloc(writerMD->get_values_size()); // This memory will be freed after the execution of the query (at callback)

        uint64_t offset=0;
        uint32_t numcolumns = writerMD->get_values()->size();
        std::cout<< "WriteTable numcols="<<numcolumns<<std::endl;
        int64_t value_size;

        for (uint32_t i=0; i < numcolumns; i++) {

            std::cout<< "WriteTable offset ="<<offset<<std::endl;
            value_size = writerMD->get_values_size(i);
            std::string value_type = ospec.getIDModelFromCol(i);

            convert_IStorage_to_UUID(((char *)cc_val)+offset, value_type, ((char*)value) + offset, value_size);

            offset += value_size;
        }

    } else if (ospec.getType() == ObjSpec::valid_types::STORAGEOBJ_TYPE) {
        if (mytype != SETATTR_TYPE) {
            throw ModuleException("IStorage:: Set Attr on a non Object is not supported");
        }
        int64_t value_size = (*writerMD->get_single_value((char*)key))[0].size;
        cc_val = malloc(value_size); // This memory will be freed after the execution of the query (at callback)

        std::string value_type = ospec.getIDModelFromColName(std::string((char*)key));
        convert_IStorage_to_UUID((char *)cc_val, value_type, value, value_size);
    }

    //std::cout << "DEBUG: IStorage::setItem: After creating value object "<<std::endl;

    // STORE THE ENTRY IN TABLE (Ex: keys + value ==> storage_id del numpy)

    std::pair<uint16_t, uint16_t> keySize = writerMD->get_keys_size();
    uint64_t partKeySize = keySize.first;
    uint64_t clustKeySize = keySize.second;
    std::cout<< "DEBUG: Istorage::setItem --> partKeySize = "<<partKeySize<<" clustKeySize = "<< clustKeySize << std::endl;

    void *cc_key= NULL;
    if (mytype == SETITEM_TYPE) {
        cc_key = malloc(partKeySize+clustKeySize); // This memory will be freed after the execution of the query (at callback)
        std::memcpy(cc_key, key, partKeySize+clustKeySize);
    } else {
        uint64_t* sid = this->getStorageID();
        void* c_key = malloc(2*sizeof(uint64_t)); //uuid
        std::memcpy(c_key, sid, 2*sizeof(uint64_t));

        cc_key = malloc(sizeof(uint64_t *)); // This memory will be freed after the execution of the query (at callback)
        std::memcpy(cc_key, &c_key, sizeof(uint64_t *));
    }

    if (mytype == SETITEM_TYPE) {
        // key arrives codified and contains latitude(double) + timestep(int)
        this->dataWriter->write_to_cassandra(cc_key, cc_val);
    } else {
        char* attr_name = (char*) key;
        this->dataWriter->write_to_cassandra(cc_key, cc_val, attr_name);
    }
}

void IStorage::setAttr(const char *attr_name, void* value) {
    /* PRE: value arrives already coded as expected: block of memory with pointers to IStorages or basic values*/
    //std::cout << "DEBUG: IStorage::setAttr: "<<std::endl;
    writeTable(attr_name, value, SETATTR_TYPE);
}

void IStorage::setAttr(const char *attr_name, IStorage* value) {
    /* 'writetable' expects a block of memory with pointers to IStorages, therefore add an indirection */
    writeTable(attr_name, (void *) &value, SETATTR_TYPE);
}

void IStorage::setItem(void* key, void* value) {
    /* PRE: value arrives already coded as expected: block of memory with pointers to IStorages or basic values*/
    writeTable(key, value, SETITEM_TYPE);
}

void IStorage::setItem(void* key, IStorage * value){
    /* 'writetable' expects a block of memory with pointers to IStorages, therefore add an indirection */
    writeTable(key, (void *) &value, SETITEM_TYPE);
}
