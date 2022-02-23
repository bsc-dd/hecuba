#include "IStorage.h"
#include <regex>

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
    uint64_t *c_uuid = currentSession->generateUUID(); // UUID for the new object
    std::string uuid = std::regex_replace(currentSession->UUID2str(c_uuid), what, "_");

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


void
IStorage::writeTable(const void* key, void* value, const enum IStorage::valid_writes mytype, const void*key_metadata, void* value_metadata) {
	/* PRE: key arrives already coded as expected */

    uint32_t value_size;

    const TableMetadata* writerMD = dataWriter->get_metadata();

    DataModel* model = this->currentSession->getDataModel();


    ObjSpec ospec = model->getObjSpec(this->id_model);
    //std::cout << "DEBUG: IStorage::setItem: obtained model for "<<id_model<<std::endl;

    std::string value_type;
    //TODO: At this moment only 1 column is supported
    if (ospec.getType() == ObjSpec::valid_types::STORAGEDICT_TYPE) {
        if (mytype != SETITEM_TYPE) {
            throw ModuleException("IStorage:: Set Item on a non Dictionary is not supported");
        }
        value_type = ospec.getIDModelFromCol(0);
    } else if (ospec.getType() == ObjSpec::valid_types::STORAGEOBJ_TYPE) {
        if (mytype != SETATTR_TYPE) {
            throw ModuleException("IStorage:: Set Attr on a non Object is not supported");
        }
        value_type = ospec.getIDModelFromColName(std::string((char*)key));
    }

    if (!ObjSpec::isBasicType(value_type)) {
        IStorage* n;
        if (value_type=="hecuba.hnumpy.StorageNumpy") {
			//if there are metadata, we assume that the value is the value to initialize a new numpy,
			//otherwise the value is the storage_id of an already existing object
			if (value_metadata != NULL) {
			    id_obj = generate_numpy_table_name(ospec.getIDObjFromCol(0)); //generate a random name based on the table name of the dictionary  and the attribute name of the value, for now hardcoded to have single-attribute values

			    // Create the numpy table:
			    // 	if the value is a StorageNumpy or a StorageObj, only the uuid is stored in the dictionary entry.
			    //	The value of the numpy/storage_obj is stored in a separated table
			    n = this->currentSession->createObject(value_type.c_str(), id_obj.c_str(), value_metadata, value);
		        value = n->getStorageID();
            }
            else {
                try {
                    value=((IStorage *)value)->getStorageID();
                } catch (std::exception &e) {
		            throw ModuleException("IStorage::set_item:  expected a wellformed StorageNumpy");
                }
            }

		} else { // An StorageObj/StorageDict...
            try {
                value=((IStorage *)value)->getStorageID();
            } catch (std::exception &e) {
	            ObjSpec ospec_value = model->getObjSpec(value_type);

	            if ((ospec_value.getType() != ObjSpec::valid_types::STORAGEDICT_TYPE)
                    && (ospec_value.getType() != ObjSpec::valid_types::STORAGEOBJ_TYPE)) {
		            throw ModuleException("IStorage:: set_item: unknow value type");
                }
                throw ModuleException("IStorage:: set_item: Expected a wellformed StorageDict or StorageObj");
            }


		}
		value_size = 2*sizeof(uint64_t);


    } else{ // it is a basic type, just copy the value
        value_size = writerMD->get_values_size();
    }
    //std::cout << "DEBUG: IStorage::setItem: After creating value object "<<std::endl;

    // STORE THE ENTRY IN TABLE (Ex: keys + value ==> storage_id del numpy)


    std::pair<uint16_t, uint16_t> keySize = writerMD->get_keys_size();
    uint64_t partKeySize = keySize.first;
    uint64_t clustKeySize = keySize.second;
    std::cout<< "DEBUG: Istorage::setItem --> partKeySize = "<<partKeySize<<" clustKeySize = "<< clustKeySize << std::endl;

    void *cc_key= NULL;
    cc_key = malloc(partKeySize+clustKeySize); //lat + ts
    if (mytype == SETITEM_TYPE) {
        std::memcpy(cc_key, key, partKeySize+clustKeySize);
    } else {
        uint64_t* sid = this->getStorageID();
        void* c_key = malloc(2*sizeof(uint64_t)); //uuid
        std::memcpy(c_key, sid, 2*sizeof(uint64_t));
        std::memcpy(cc_key, &c_key, partKeySize+clustKeySize);
    }


    //std::cout<< "DEBUG: Istorage::setItem --> value" << value << std::endl;

    void * cc_val;
    if (!ObjSpec::isBasicType(value_type)) {
        uint64_t* c_value_copy = (uint64_t*)malloc(value_size);
        std::memcpy(c_value_copy, value, value_size);
        cc_val = malloc(sizeof(uint64_t*)); //uuid(numpy)
        std::memcpy((char *)cc_val, &c_value_copy, sizeof(uint64_t*));
    }else {
        cc_val = (uint64_t*)malloc(value_size);
        std::memcpy(cc_val, value, value_size);
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
    /* PRE: value arrives already coded as expected */
    //std::cout << "DEBUG: IStorage::setAttr: "<<std::endl;
    writeTable(attr_name, value, SETATTR_TYPE);
}

void IStorage::setAttr(const char *attr_name, IStorage* value) {
    /* PRE: key arrives already coded as expected */
    writeTable(attr_name, (void *) value, SETATTR_TYPE);
}

void IStorage::setItem(void* key, void* value, void *key_metadata, void *value_metadata) {
    /* PRE: key arrives already coded as expected */
    //std::cout << "DEBUG: IStorage::setItem: "<<std::endl;
    writeTable(key, value, SETITEM_TYPE, key_metadata, value_metadata);
}

void IStorage::setItem(void* key, IStorage * value){
    /* PRE: key arrives already coded as expected */
    //std::cout << "DEBUG: IStorage::setItem: "<<std::endl;
    writeTable(key, (void *) value, SETITEM_TYPE);
}
