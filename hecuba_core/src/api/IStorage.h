#ifndef ISTORAGE_H
#define ISTORAGE_H


#include "configmap.h"
#include "HecubaSession.h"

//class HecubaSession; //Forward declaration

class IStorage {
    /* This class represents an instantiated object (StorageDict) and can be
     * used as a gateway to modify the associated table. */

public:
    IStorage(HecubaSession* session, std::string id_model, std::string id_object, uint64_t* storage_id, Writer* writer);
    ~IStorage();

    //void setItem(void* keys, void* values); // If all type information is known from the Model
    void setItem(void* keys, void* values, void* keyMetadata=NULL, void* valueMetadata=NULL);

    void setAttr(const std::string& attr_name, void* value);

	uint64_t* getStorageID();

    Writer * getDataWriter();

    void sync(void);

private:
	void decodeNumpyMetadata(HecubaSession::NumpyShape* s, void* metadata);
    std::string generate_numpy_table_name(std::string attributename);

    config_map keysnames;
    config_map keystypes;
    config_map colsnames;
    config_map colstypes;

    uint64_t* storageid;

	std::string id_obj; // Name to identify this 'object' [keyspace.name]
	std::string id_model; // Type name to be found in model "class_name"

	HecubaSession* currentSession; //Access to cassandra and model

	Writer* dataWriter; /* Writer for entries in the object */
};
#endif /* ISTORAGE_H */
