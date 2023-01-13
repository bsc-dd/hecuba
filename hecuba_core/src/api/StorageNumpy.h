#ifndef _STORAGENUMPY_
#define _STORAGENUMPY_

#include <map>
#include <iostream>
#include <type_traits>
#include <hecuba/ObjSpec.h>
#include <hecuba/debug.h>
#include <hecuba/IStorage.h>
#include <hecuba/KeyClass.h>
#include <hecuba/ValueClass.h>
#include "UUID.h"
#include "ArrayDataStore.h"


#include "SpaceFillingCurve.h"
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

class StorageNumpy:public IStorage {
public:

    void *data = nullptr;   /* Pointer to memory containing the object. */
    std::vector<uint32_t> metas;
    ArrayMetadata numpy_metas; /* Pointer to memory containing the metadata. READ ONLY. DO NOT FREE. This object does NOT own the memory! */
    void initObjSpec() {
	    std::vector<std::pair<std::string, std::string>> pkeystypes_numpy = {
		    {"storage_id", "uuid"},
		    {"cluster_id", "int"}
	    };
	    std::vector<std::pair<std::string, std::string>> ckeystypes_numpy = {
		    {"block_id","int"}
	    };
	    std::vector<std::pair<std::string, std::string>> colstypes_numpy = {
		    {"payload", "blob"},
	    };
	    ObjSpec snSpec = ObjSpec(
			    ObjSpec::valid_types::STORAGENUMPY_TYPE
			    , pkeystypes_numpy
			    , ckeystypes_numpy
			    , colstypes_numpy
			    , std::string("")
			    );
	    setObjSpec(snSpec);
    }

    StorageNumpy() {
	initObjSpec();
    }

    StorageNumpy(void *datasrc, const std::vector<uint32_t> &metas) {
	// Transform user metas to ArrayMetadata
	this->metas = metas; // make a copy of user 'metas'
	uint32_t numpy_size = getMetaData(metas, this->numpy_metas);

	// Make a copy of 'datasrc'
	this->data = malloc(numpy_size);
	memcpy(this->data, datasrc, numpy_size);

	initObjSpec();
    }
 
    // StorageNumpy sn = misn;
    StorageNumpy(StorageNumpy &src) {
	StorageNumpy(src.data, src.metas);
    }

    // StorageNumpy sn; sn = misn;
    StorageNumpy &operator = (const StorageNumpy & w) {
	this->metas = w.metas;
	uint32_t numpy_size = getMetaData(metas, this->numpy_metas);
	this->data = malloc(numpy_size);
	memcpy(this->data, w.data, numpy_size);
    }
    
    ~StorageNumpy() {
	if (this->data != nullptr) {
		free (this->data);
	}
	#if 0
	if (this->arrayStore != nullptr) {
		delete (this->arrayStore); //this causes the deletion of the other two caches and when the destructor of the IStorage is called it fails
	}
	#endif
    }

    // this function is called in the destructor of IStorage;
    void deleteCache() {
	if (this->arrayStore != nullptr) {
		delete (this->arrayStore);
	}
    }

    void assignTableName(std::string id_obj, std::string id_model) {
	this->setTableName(id_obj); //in the case of StorageNumpy this will be the name of the class
    }

    void initialize_dataAcces() {
	    // StorageNumpy
	    HecubaSession *currentSession = getCurrentSession(); 
	    this->arrayStore = new ArrayDataStore(getTableName().c_str(), 
	    		currentSession->config["execution_name"].c_str(),
			currentSession->getStorageInterface()->get_session(), currentSession->config);
	    this->setCache( this->arrayStore->getWriteCache() );
    }

    void persist_metadata(uint64_t* c_uuid) {
	    // Register in hecuba.istorage
	    registerNumpy(this->numpy_metas,  getObjectName(), c_uuid);
    }

    void persist_data() {
	    // Dump data into Cassandra
	    arrayStore->store_numpy_into_cas(getStorageID(), this->numpy_metas, this->data);

    }

void send(void) {
    DBG("DEBUG: IStorage::send: sending numpy. Size "<< numpy_metas.get_array_size());
    getDataWriter()->send_event((char *) data, numpy_metas.get_array_size());
}

void writePythonSpec() {} // StorageNumpy do not have python specification

private:

	ArrayDataStore* arrayStore = nullptr; /* Cache of written/read elements */

    uint32_t getMetaData(const std::vector<uint32_t> &raw_numpy_meta, ArrayMetadata &arr_metas) {
    	std::vector <uint32_t> dims;
    	std::vector <uint32_t> strides;

    	// decode void *metadatas
    	uint32_t acum=1;
    	uint32_t numpy_size=0;
    	for (uint32_t i=0; i < raw_numpy_meta.size(); i++) {
        	dims.push_back( raw_numpy_meta[i]);
        	acum *= raw_numpy_meta[i];
    	}
	numpy_size = acum;
    	for (uint32_t i=0; i < raw_numpy_meta.size(); i++) {
        	strides.push_back(acum * sizeof(double));
        	acum /= raw_numpy_meta[raw_numpy_meta.size()-1-i];
    	}
    	uint32_t flags=NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED;
	
    	arr_metas.dims = dims;
    	arr_metas.strides = strides;
    	arr_metas.elem_size = sizeof(double);
    	arr_metas.flags = flags;
    	arr_metas.partition_type = ZORDER_ALGORITHM;
    	arr_metas.typekind = 'f';
    	arr_metas.byteorder = '=';
	return numpy_size;
    }

    void registerNumpy(ArrayMetadata &numpy_meta, std::string name, uint64_t* uuid) {

	//std::cout<< "DEBUG: HecubaSession::registerNumpy BEGIN "<< name << UUID::UUID2str(uuid)<<std::endl;
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
		getCurrentSession()->getNumpyMetaWriter()->write_to_cassandra(keys, values);
		getCurrentSession()->getNumpyMetaWriter()->wait_writes_completion(); // Ensure hecuba.istorage get all updates SYNCHRONOUSLY (to avoid race conditions with poll that may request a build_remotely on this new object)!
	}
	catch (std::exception &e) {
		std::cerr << "HecubaSession::registerNumpy: Error writing" <<std::endl;
		std::cerr << e.what();
		throw e;
	};

}
    
};

#endif /* _STORAGENUMPY_ */
