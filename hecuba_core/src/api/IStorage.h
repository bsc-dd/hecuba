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

    void setItem(void* key, IStorage* value);
    void setItem(void* keys, void* values);

    void send(void* key, IStorage* value);
    void send(void* key, void* value);
    void send(void);

    void setAttr(const char* attr_name, IStorage* value);
    void setAttr(const char* attr_name, void* value);

	uint64_t* getStorageID();

    Writer * getDataWriter();

    void sync(void);

    void setNumpyAttributes(ArrayMetadata &metas, void* value);

    void enableStream(std::string topic);
    bool isStream();

private:
    enum valid_writes {
        SETATTR_TYPE,
        SETITEM_TYPE,
    };
    void writeTable(const void* key, void* value, const enum valid_writes mytype);
    std::string generate_numpy_table_name(std::string attributename);

    /* convert_IStorage_to_UUID: Given a value (basic or persistent) convert it to the same value or its *storage_id* if it is a persistent one. Returns True if type is converted (aka was an IStorage). */
    bool convert_IStorage_to_UUID(char * dst, const std::string& value_type, void* src, int64_t src_size) const ;

    config_map keysnames;
    config_map keystypes;
    config_map colsnames;
    config_map colstypes;

    uint64_t* storageid;

	std::string id_obj; // Name to identify this 'object' [keyspace.name]
	std::string id_model; // Type name to be found in model "class_name"

	HecubaSession* currentSession; //Access to cassandra and model

    bool streamEnabled=false;

	Writer* dataWriter; /* Writer for entries in the object */

    void *data;   /* Pointer to memory containing the object. READ ONLY. DO NOT FREE. This object does NOT own the memory! */
    ArrayMetadata numpy_metas; /* Pointer to memory containing the metadata. READ ONLY. DO NOT FREE. This object does NOT own the memory! */
};
#endif /* ISTORAGE_H */
