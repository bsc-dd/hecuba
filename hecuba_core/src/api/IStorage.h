#ifndef ISTORAGE_H
#define ISTORAGE_H


#include "configmap.h"
#include "HecubaSession.h"
#include "ArrayDataStore.h"
#include "debug.h"

//class HecubaSession; //Forward declaration

#define KEYS    1
#define COLUMNS 0

class IStorage {
    /* This class represents an instantiated object (StorageDict) and can be
     * used as a gateway to modify the associated table. */

public:
    IStorage();
    IStorage(HecubaSession* session, std::string id_model, std::string id_object, uint64_t* storage_id, CacheTable* reader);
    ~IStorage();

    void setItem(void* key, IStorage* value);
    void setItem(void* keys, void* values);

    void send(void* key, IStorage* value);
    void send(void* key, void* value);
    void send(void);
    void send_values(const void* value);

    void setAttr(const char* attr_name, IStorage* value);
    void setAttr(const char* attr_name, void* value);

    void setClassName(std::string name);
    void setIdModel(std::string name);
    std::string getClassName();
    void setSession(HecubaSession *s);
    HecubaSession * getCurrentSession();
    const std::string& getTableName() const;
    void setTableName(std::string tableName);
    void make_persistent(const std::string id_obj);
    bool is_pending_to_persist();

    void getAttr(const char* attr_name, void * valuetoreturn) const;
    void getItem(const void* key, void * valuetoreturn) const;

	uint64_t* getStorageID();
	const std::string& getName() const;

    Writer * getDataWriter();

    void sync(void);

    void enableStream(std::string topic);
    bool isStream();

    void setNumpyAttributes(ArrayDataStore * array_store, ArrayMetadata &metas, void* value=NULL);
    void * getNumpyData() const;

    /* Iterators */
    struct keysIterator {
        using iterator_category = std::input_iterator_tag;
        using difference_type   = std::ptrdiff_t;   // Is it really needed?
        //using value_type        = TupleRow;
        using pointer           = TupleRow*;  // or also value_type*
        using reference         = void*;  // or also value_type&

        // Constructor
        keysIterator(void) : m_ptr(nullptr) {}
        //keysIterator(pointer ptr) : m_ptr(ptr) {}
        //keysIterator(IStorage *my_instance, Prefetch *my_P) : instance(my_instance), P(my_P) {m_ptr = P->get_cnext();}
        keysIterator(IStorage *my_instance, Prefetch *my_P) {P = my_P; instance = my_instance; m_ptr = P->get_cnext(); DBG(" m_ptr == " << (uint64_t)m_ptr);DBG( " PAYLOAD == "<<(int64_t)m_ptr->get_payload());}

        // Operators
        reference operator*() const {
            /* TODO return the payload from the current TupleRow (processing it before) */
            void *valueToReturn=nullptr;
            //instance->extractMultiValuesFromQueryResult(m_ptr->get_payload(), valueToReturn);
            DBG(" PRINT INOFENSIVO...");
            DBG(" m_ptr == " << (uint64_t)m_ptr);
            DBG(" BEFORE Get Payload...");
            void *valueFromQuery = m_ptr->get_payload();
            DBG(" BEFORE Extracting values...");
            instance->extractMultiValuesFromQueryResult(valueFromQuery, &valueToReturn, KEYS);
            return valueToReturn;
        }
        //pointer operator->() {
        //    /* TODO: return the pointer to the processed payload from the current TupleRow */
        //     return m_ptr;
        //}

        // Prefix increment
        keysIterator& operator++() {
            m_ptr = P->get_cnext();
            return *this;
        }
        // Postfix increment
        keysIterator operator++(int) { keysIterator tmp = *this; ++(*this); return tmp; }

        friend bool operator== (const keysIterator& a, const keysIterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!= (const keysIterator& a, const keysIterator& b) { return a.m_ptr != b.m_ptr; };


    private:

        pointer m_ptr;
        Prefetch *P;
        IStorage *instance;
    };

    //struct valuesIterator; /* These may need some hacking */
    //struct itemsIterator;

    keysIterator begin();
    keysIterator end();
    void writePythonSpec();
    ObjSpec getObjSpec();
    void setObjSpec(ObjSpec oSpec);
    void setPythonSpec(std::string pSpec);
    std::string getPythonSpec();
    void setObjectName(std::string id_obj);
    std::string getTableName();
    // the definition of at least one virtual function is necessary to use rtti capabilities
    // and be able to infer subclass name from a method in the base clase
    virtual void generatePythonSpec() {};
    virtual void assignTableName(std::string id_object, std::string id_model) {};
    virtual void persist_metadata(uint64_t * c_uuid) {};
    virtual std::vector<std::pair<std::string, std::string>> getValuesDesc() {};


private:
    ObjSpec IStorageSpec;
    std::string pythonSpec = "";
    std::string tableName;
    bool pending_to_persist = false;
    bool persistent = false;

    enum valid_writes {
        SETATTR_TYPE,
        SETITEM_TYPE,
    };
    void writeTable(const void* key, const void* value, const enum valid_writes mytype);
    std::string generate_numpy_table_name(std::string attributename);

    /* convert_IStorage_to_UUID: Given a value (basic or persistent) convert it to the same value or its *storage_id* if it is a persistent one. Returns True if type is converted (aka was an IStorage). */
    bool convert_IStorage_to_UUID(char * dst, const std::string& value_type, const void* src, int64_t src_size) const ;
    void * deep_copy_attribute_buffer(bool isKey, const void* src, uint64_t src_size, uint32_t num_attrs) const ;

    config_map keysnames;
    config_map keystypes;
    config_map colsnames;
    config_map colstypes;

    uint64_t* storageid;

	std::string id_obj=""; // Name to identify this 'object' [keyspace.name]
	std::string id_model=""; // Type name to be found in model "class_name" (FQName)
	std::string class_name=""; // plain class name

	HecubaSession* currentSession; //Access to cassandra and model

    bool streamEnabled=false;

	Writer* dataWriter; /* Writer for entries in the object. EXTRACTED from 'dataAccess' */
	CacheTable* dataAccess; /* Cache of written/read elements */
	ArrayDataStore* arrayStore; /* Cache of written/read elements */

    void *data;   /* Pointer to memory containing the object. READ ONLY. DO NOT FREE. This object does NOT own the memory! */
    ArrayMetadata numpy_metas; /* Pointer to memory containing the metadata. READ ONLY. DO NOT FREE. This object does NOT own the memory! */

    void extractFromQueryResult(std::string value_type, uint32_t value_size, void *query_result, void *valuetoreturn) const;
    void extractMultiValuesFromQueryResult(void *query_result, void *valuetoreturn, int type) const ;
};
#endif /* ISTORAGE_H */
