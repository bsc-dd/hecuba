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
    std::cout << "DEBUG: IStorage::generate_numpy_table_name: BEGIN attribute:"<<attributename<<std::endl;
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
    std::cout << "DEBUG: IStorage::generate_numpy_table_name: END "<<name<<std::endl;
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
    std::cout << "DEBUG: IStorage::setItem: "<<std::endl;
	DataModel* model = this->currentSession->getDataModel();

	DataModel::obj_spec ospec = model->getObjSpec(this->id_model);
    std::cout << "DEBUG: IStorage::setItem: obtained model for "<<id_model<<std::endl;

	if (ospec.objtype != DataModel::STORAGEDICT_TYPE) {
		throw ModuleException("IStorage:: Only Dictionary are supported");
	}

	// TODO Check the key type. NOW HARDCODED to int, int (HecubaSession::loadDataModel)
	// TODO Check the value type. NOW HARDCODED TO Numpy (HecubaSession::loadDataModel)
	HecubaSession::NumpyShape* valMD = NULL;;
    if (value_metadata != NULL) {
        valMD = new HecubaSession::NumpyShape();
	    decodeNumpyMetadata(valMD, value_metadata);
        //std::cout << "DEBUG: IStorage::setItem: MetaData for numpy value decoded with "<<valMD->ndims<< " dims:" <<valMD->debug()<<std::endl;
    }

	// Crear Numpy

    id_model = ospec.cols[0].second; //'hecuba.hnumpy.StorageNumpy'
    id_obj = generate_numpy_table_name(ospec.cols[0].first); //genera a random name based on the table name of the dictionary  and the attribute name of the value, for now hardcoded to have single-attribute values

       //crear tabla numpy
    IStorage* n = this->currentSession->createObject(id_model.c_str(), id_obj.c_str(), valMD, value);

    //std::cout << "DEBUG: IStorage::setItem: After creating value object "<<std::endl;

    // GUARDA LA ENTRADA DEL DICCIONARIO (keys + value==> storage_id del numpy)


    const TableMetadata* writerMD = dataWriter->get_metadata();

    std::pair<uint16_t, uint16_t> keySize = writerMD->get_keys_size();
    uint64_t partKeySize = keySize.first;
    uint64_t clustKeySize = keySize.second;
    //std::cout<< "DEBUG: Istorage::setItem --> partKeySize = "<<partKeySize<<" clustKeySize = "<< clustKeySize << std::endl;


    void * cc_key = malloc(partKeySize+clustKeySize); //lat + ts
    std::memcpy(cc_key, key, partKeySize+clustKeySize);


    // Copy UUID
	uint64_t* c_uuid = n->getStorageID();
    //std::cout<< "DEBUG: Istorage::setItem --> generated UUID" << currentSession->UUID2str(c_uuid) << std::endl;
    uint64_t* c_uuid_copy = (uint64_t*)malloc(sizeof(uint64_t)*2);
    std::memcpy(c_uuid_copy, c_uuid, sizeof(uint64_t)*2);
    //std::cout<< "DEBUG: Istorage::setItem --> &UUID" << c_uuid_copy << std::endl;

    void * cc_val = malloc(sizeof(uint64_t*)); //uuid(numpy)
    std::memcpy((char *)cc_val, &c_uuid_copy, sizeof(uint64_t*));


    // key arrives codified and contains latitude(double) + timestep(int)
	this->dataWriter->write_to_cassandra(cc_key, cc_val);
}
