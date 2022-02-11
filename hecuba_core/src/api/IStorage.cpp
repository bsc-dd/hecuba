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

void IStorage::decodeNumpyMetadata(HecubaSession::NumpyShape *s, void* metadata) {
	// Numpy Metadata(all unsigned): Ndims + Dim1 + Dim2 + ... + DimN  (Layout in C by default)
	unsigned* value = (unsigned*)metadata;
	s->ndims = *(value);
	value ++;
	s->dim = (unsigned *)malloc(s->ndims);
    for (unsigned i = 0; i < s->ndims; i++) {
		s->dim[i] = *(value + i);
	}
}

uint64_t* IStorage::getStorageID() {
    return storageid;
}

void IStorage::setItem(void* key, void* value, void *key_metadata, void *value_metadata) {
	/* PRE: key arrives already coded as expected */
	//std::cout << "DEBUG: IStorage::setItem: "<<std::endl;
	uint32_t value_size;

	const TableMetadata* writerMD = dataWriter->get_metadata();

	DataModel* model = this->currentSession->getDataModel();


	ObjSpec ospec = model->getObjSpec(this->id_model);
	//std::cout << "DEBUG: IStorage::setItem: obtained model for "<<id_model<<std::endl;

	if (ospec.getType() != ObjSpec::valid_types::STORAGEDICT_TYPE) {
		throw ModuleException("IStorage:: Only Dictionary are supported");
	}

    //TODO: At this moment only 1 column is supported
	std::string value_type = ospec.getIDModelFromCol(0);//'hecuba.hnumpy.StorageNumpy'

	if (!ObjSpec::isBasicType(value_type)) {
		IStorage* n;
		if (value_type=="hecuba.hnumpy.StorageNumpy") {
			HecubaSession::NumpyShape* valMD = NULL;;
			if (value_metadata == NULL) {
				throw ModuleException("IStorage:: setItem with a Numpy, but Metadata is missing.");
			}
			valMD = new HecubaSession::NumpyShape();
			decodeNumpyMetadata(valMD, value_metadata);
			//std::cout << "DEBUG: IStorage::setItem: MetaData for numpy value decoded with "<<valMD->ndims<< " dims:" <<valMD->debug()<<std::endl;
			id_obj = generate_numpy_table_name(ospec.getIDObjFromCol(0)); //genera a random name based on the table name of the dictionary  and the attribute name of the value, for now hardcoded to have single-attribute values

			// Create the numpy table:
			// 	if the value is a StorageNumpy or a StorageObj, only the uuid is stored in the dictionary entry.
			//	The value of the numpy/storage_obj is stored in a separated table
			n = this->currentSession->createObject(value_type.c_str(), id_obj.c_str(), valMD, value);

		} else {
			throw ModuleException("IStorage:: setItem with StorageObj NOT SUPPORTED YET");
			// Si es un storage object crear el objeto igual pero sin metadatos.
			// Hay que anyadir el case al create object, pero sera como el del diccionario. Pero en el caso del
			// storageobj comparten tabla todos los storage obj... asi que la query de create table tiene que ser
			// create if not exists

		}
		value = n->getStorageID();
		value_size = 2*sizeof(uint64_t);

	} else{ // it is a basic type, just copy the value
		value_size = writerMD->get_values_size();
	}
	//std::cout << "DEBUG: IStorage::setItem: After creating value object "<<std::endl;

	// GUARDA LA ENTRADA DEL DICCIONARIO (keys + value==> storage_id del numpy)


	std::pair<uint16_t, uint16_t> keySize = writerMD->get_keys_size();
	uint64_t partKeySize = keySize.first;
	uint64_t clustKeySize = keySize.second;
	std::cout<< "DEBUG: Istorage::setItem --> partKeySize = "<<partKeySize<<" clustKeySize = "<< clustKeySize << std::endl;


	void * cc_key = malloc(partKeySize+clustKeySize); //lat + ts
	std::memcpy(cc_key, key, partKeySize+clustKeySize);


	//std::cout<< "DEBUG: Istorage::setItem --> value" << value << std::endl;
	uint64_t* c_value_copy = (uint64_t*)malloc(value_size);
	std::memcpy(c_value_copy, value, value_size);

	void * cc_val = malloc(sizeof(uint64_t*)); //uuid(numpy)
	std::memcpy((char *)cc_val, &c_value_copy, sizeof(uint64_t*));


	// key arrives codified and contains latitude(double) + timestep(int)
	this->dataWriter->write_to_cassandra(cc_key, cc_val);
}

Writer *
IStorage::getDataWriter(void) {
    return dataWriter;
}

void
IStorage::sync(void) {
    this->dataWriter->flush_elements();
}

void IStorage::setAttr(const std::string &attr_name, void* value) {
	/* PRE: value arrives already coded as expected */
	//std::cout << "DEBUG: IStorage::setItem: "<<std::endl;
	uint32_t value_size;

	const TableMetadata* writerMD = dataWriter->get_metadata();

	DataModel* model = this->currentSession->getDataModel();


	ObjSpec ospec = model->getObjSpec(this->id_model);
	//std::cout << "DEBUG: IStorage::setItem: obtained model for "<<id_model<<std::endl;

	if (ospec.getType() != ObjSpec::valid_types::STORAGEOBJ_TYPE) {
		throw ModuleException("IStorage:: Only StorageObj are supported by setAttr");
	}

	std::string value_type = ospec.getIDModelFromColName(attr_name);//'hecuba.hnumpy.StorageNumpy'

	if (!ObjSpec::isBasicType(value_type)) {
		throw ModuleException("IStorage:: setAttr with persistent objects NOT IMPLEMENTED.");
            /* TODO: Best option should be to receive anthor IStorage
		IStorage* n;
		if (value_type=="hecuba.hnumpy.StorageNumpy") {
			HecubaSession::NumpyShape* valMD = NULL;;
			if (value_metadata == NULL) {
				throw ModuleException("IStorage:: setItem with a Numpy, but Metadata is missing.");
			}
			valMD = new HecubaSession::NumpyShape();
			decodeNumpyMetadata(valMD, value_metadata);
			//std::cout << "DEBUG: IStorage::setItem: MetaData for numpy value decoded with "<<valMD->ndims<< " dims:" <<valMD->debug()<<std::endl;
			id_obj = generate_numpy_table_name(ospec.getIDObjFromCol(0)); //genera a random name based on the table name of the dictionary  and the attribute name of the value, for now hardcoded to have single-attribute values

			// Create the numpy table:
			// 	if the value is a StorageNumpy or a StorageObj, only the uuid is stored in the dictionary entry.
			//	The value of the numpy/storage_obj is stored in a separated table
			n = this->currentSession->createObject(value_type.c_str(), id_obj.c_str(), valMD, value);

		} else {
			throw ModuleException("IStorage:: setItem with StorageObj NOT SUPPORTED YET");
			// Si es un storage object crear el objeto igual pero sin metadatos.
			// Hay que anyadir el case al create object, pero sera como el del diccionario. Pero en el caso del
			// storageobj comparten tabla todos los storage obj... asi que la query de create table tiene que ser
			// create if not exists

		}
		value = n->getStorageID();
		value_size = 2*sizeof(uint64_t);
            */

	} else{ // it is a basic type, just copy the value
		value_size = writerMD->get_values_size();
	}
	//std::cout << "DEBUG: IStorage::setItem: After creating value object "<<std::endl;

	// GUARDA LA ENTRADA DEL OBJETO (storage_id).attr_name=value)


	std::pair<uint16_t, uint16_t> keySize = writerMD->get_keys_size();
	uint64_t partKeySize = keySize.first;   // This should be sizeof(uuid)
	uint64_t clustKeySize = keySize.second; // This should be zero
	std::cout<< "DEBUG: Istorage::setAttr --> partKeySize = "<<partKeySize<<" clustKeySize = "<< clustKeySize << std::endl;

    uint64_t* key = this->getStorageID();
    // Make a copy of StorageID
    void* c_key = malloc(2*sizeof(uint64_t)); //uuid
    std::memcpy(c_key, key, 2*sizeof(uint64_t));

    // Prepare space for the pointer to the StorageID copy
	void * cc_key = malloc(partKeySize+clustKeySize);
    //yolandab
	//std::memcpy(cc_key, key, partKeySize+clustKeySize);
	std::memcpy(cc_key, &c_key, partKeySize+clustKeySize);


	//std::cout<< "DEBUG: Istorage::setItem --> value" << value << std::endl;
	uint64_t* c_value_copy = (uint64_t*)malloc(value_size);
	std::memcpy(c_value_copy, value, value_size);


	// key arrives codified and contains latitude(double) + timestep(int)
	this->dataWriter->write_to_cassandra(cc_key, c_value_copy, attr_name.c_str());
}
