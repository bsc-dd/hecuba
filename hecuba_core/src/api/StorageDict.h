#ifndef _STORAGEDICT_
#define _STORAGEDICT_

#include <map>
#include <iostream>
#include <type_traits>
#include <hecuba/ObjSpec.h>
#include <hecuba/debug.h>
#include <hecuba/IStorage.h>
#include <hecuba/KeyClass.h>
#include <hecuba/ValueClass.h>
#include "UUID.h"
#include <hecuba/StorageNumpy.h>


template<class K, class V>

class StorageDict:virtual public IStorage {

#define ISKEY true

public:
    void initObjSpec() {
	K key;
	V v;
	partitionKeys=key.getPartitionKeys();
	clusteringKeys=key.getClusteringKeys();
	valuesDesc=v.getValuesDesc("valuename");
	ObjSpec dictSpec;
	dictSpec=ObjSpec(ObjSpec::valid_types::STORAGEDICT_TYPE, partitionKeys, clusteringKeys, valuesDesc,"");
 	setObjSpec(dictSpec);
    }

    StorageDict() {
    	initObjSpec();
    }

    // c++ only calls implicitly the constructor without parameters. To invoke this constructor we need to add to the user class an explicit call to this
    StorageDict(const std::string name) {
	initObjSpec();
	id_obj=name;
	pending_to_persist=true;
    }
    
    ~StorageDict() {}


    // sd[k] = v or v = sd[k] 
    // return a reference to allow sd[k]=v
    // in the operator = for ValueClass we determine in which case we are
    V& operator[](K &key) {
	V *v = new V(this, key.getKeysBuffer(),key.getTotalSize());
        return *v;
    }


    // It generates the python specification for the class during the registration of the object
    void generatePythonSpec() {
	std::string StreamPart="";
	if (isStream() ){
		StreamPart=std::string(", StorageStream");
	}
	std::string pythonSpec = "from hecuba import StorageDict"
				  + StreamPart +
				  + "\n\nclass "
				  + getClassName() + "(StorageDict"
				  + StreamPart
				  + "):\n"
				  + "   '''\n   @TypeSpec dict <<";

	std::string itemSpec = "";
	for (std::vector<std::pair<std::string,std::string>>::iterator it=partitionKeys.begin(); it!=partitionKeys.end(); ++it)
		itemSpec+=it->first + ":"+ it->second + ",";
	for (std::vector<std::pair<std::string,std::string>>::iterator it=clusteringKeys.begin(); it!=clusteringKeys.end(); ++it)
		itemSpec+=it->first + ":"+ it->second + ",";

	pythonSpec += itemSpec.substr(0, itemSpec.size()-1) + ">,"; // replace the last , with a >

	itemSpec = "";
	for (std::vector<std::pair<std::string,std::string>>::iterator it=valuesDesc.begin(); it!=valuesDesc.end(); ++it)
		itemSpec+=it->first + ":"+ it->second + ",";

	pythonSpec += itemSpec.substr(0, itemSpec.size()-1) + ">\n   '''\n"; // replace the last , with a >

	setPythonSpec(pythonSpec);
    }

    void assignTableName(const std::string& id_obj, const std::string& id_model) {
        size_t pos = id_obj.find_first_of(".");
        std::string tablename =id_obj.substr(pos+1, id_obj.size());
        this->setTableName( tablename ); //in the case of StorageObject this will be the name of the class
    }

    void persist_metadata(uint64_t* c_uuid) {
	ObjSpec oType = getObjSpec(); 
    	std::string insquery = 	std::string("INSERT INTO ") +
                        	std::string("hecuba.istorage") +
                        	std::string("(storage_id, name, class_name, primary_keys, columns)") +
                        	std::string("VALUES ") +
                        	std::string("(") +
                        	UUID::UUID2str(c_uuid) + std::string(", ") +
                        	"'" + getCurrentSession()->config["execution_name"] + "." + getTableName() + "'" + std::string(", ") +
                        	"'" + this->getIdModel() + "'" + std::string(", ") +
                        	oType.getKeysStr() + std::string(", ") +
                        	oType.getColsStr() +
                        	std::string(")");
        CassError rc = getCurrentSession()->run_query(insquery);
                if (rc != CASS_OK) {
                    std::string msg = std::string("StorageDict::persist_metadata: Error executing query ") + query;
                    throw ModuleException(msg);
                }
    }

	/* setPersistence - Inicializes current instance to conform to uuid object. To be used on an empty instance. */
    void setPersistence (uint64_t *uuid) {
	    // FQid_model: Fully Qualified name for the id_model: module_name.id_model
	    std::string FQid_model = this->getIdModel();

	    struct metadata_info row = this->getMetaData(uuid);

	    std::pair<std::string, std::string> idmodel = getKeyspaceAndTablename( row.name );
	    std::string keyspace = idmodel.first;
	    std::string tablename = idmodel.second;

	    const char * id_object = tablename.c_str();
	    // Check that retrieved classname form hecuba coincides with 'id_model'
	    std::string sobj_table_name = row.class_name;

	    // The class_name retrieved in the case of the storageobj is
	    // the fully qualified name, but in cassandra the instances are
	    // stored in a table with the name of the last part(example:
	    // "model_complex.info" will have instances in "keyspace.info")
	    // meaning that in a complex scenario with different models...
	    // we will loose information. FIXME
	    if (sobj_table_name.compare(FQid_model) != 0) {
		    throw ModuleException("HecubaSession::createObject uuid "+UUID::UUID2str(uuid)+" "+ tablename + " has unexpected class_name " + sobj_table_name + " instead of "+FQid_model);
	    }

	    init_persistent_attributes(tablename, uuid);
	    // Create READ/WRITE cache accesses
	    initialize_dataAcces();


    }


    std::vector<std::pair<std::string, std::string>> getValuesDesc () {
	return valuesDesc;
    }


    void setItem( void *key, void *value) {

	void * cc_val;
        const TableMetadata* writerMD = getDataWriter()->get_metadata();
	// prepare values
	std::shared_ptr<const std::vector<ColumnMeta> > columns = writerMD->get_values();
        uint32_t numcolumns = columns->size();
        cc_val = deep_copy_attribute_buffer(!ISKEY, value, writerMD->get_values_size(), numcolumns);
	// prepare keys
	std::pair<uint16_t, uint16_t> keySize = writerMD->get_keys_size();
	uint64_t partKeySize = keySize.first;
	uint64_t clustKeySize = keySize.second;
	DBG("IStorage::writeTable --> partKeySize = "<<partKeySize<<" clustKeySize = "<< clustKeySize);
	void *cc_key= NULL;
	columns = writerMD->get_keys();
        numcolumns = columns->size();
        cc_key = deep_copy_attribute_buffer(ISKEY, key, partKeySize+clustKeySize, numcolumns);

        const TupleRow* trow_key = this->getDataAccess()->get_new_keys_tuplerow(cc_key);
        const TupleRow* trow_values = this->getDataAccess()->get_new_values_tuplerow(cc_val);
#if 1
        if (this->isStream()) {
            this->getDataWriter()->send_event(trow_key, trow_values); // stream value (storage_id/value)
            send_values(value); // If value is an IStorage type stream its contents also
        }
#endif
        this->getDataAccess()->put_crow(trow_key, trow_values);
        delete(trow_key);
        delete(trow_values);
    }

void send_values(const void *value) {
    DBG("START");
    const TableMetadata* writerMD = getDataWriter()->get_metadata();
    ObjSpec ospec = this->getObjSpec();

    std::shared_ptr<const std::vector<ColumnMeta> > columns = writerMD->get_values();
    uint32_t numcolumns = columns->size();

    uint64_t offset = 0;
    const char* src = (char*)value;
    // Traverse the buffer following the user order...
    for (uint32_t i=0; i < numcolumns; i++) {
        std::string column_name = ospec.getIDObjFromCol(i);
        std::string value_type = ospec.getIDModelFromCol(i);
        const ColumnMeta *c = writerMD->get_single_column(column_name);
        int64_t value_size= c->size;
        DBG(" -->  traversing column '"<<column_name<< "' of type '" << value_type<<"'" );
        if (!ObjSpec::isBasicType(value_type)) {
            if (value_type.compare("hecuba.hnumpy.StorageNumpy") == 0) {
                StorageNumpy * result = *(StorageNumpy **)(src+offset); // 'src' MUST be a valid pointer or it will segfault here...
                if (!result->isStream()) { // If the object did not have Stream enabled, enable it now as we are going to stream it...
                    result->configureStream(UUID::UUID2str(result->getStorageID()));
                }
                result->send();
                DBG("   -->  sent "<< UUID::UUID2str(result->getStorageID()));
            }
        }
        offset += value_size;
    }
    DBG("END");
}

void initialize_dataAcces() {
        //  Create Writer
        ObjSpec oType = this->getObjSpec();
        std::vector<config_map>* keyNamesDict = oType.getKeysNamesDict();
        std::vector<config_map>* colNamesDict = oType.getColsNamesDict();
        CacheTable *reader = getCurrentSession()->getStorageInterface()->make_cache(this->getTableName().c_str(),
                                getCurrentSession()->config["execution_name"].c_str(), *keyNamesDict, *colNamesDict, getCurrentSession()->config);
	this->setCache(reader);

        delete keyNamesDict;
        delete colNamesDict;
}

std::vector<std::pair<std::string, std::string>> getPartitionKeys(){
	return partitionKeys;
}
std::vector<std::pair<std::string, std::string>> getClusteringKeys(){
	return clusteringKeys;
}


    /* Iterators */
struct keysIterator {
        using iterator_category = std::input_iterator_tag;
        using difference_type   = std::ptrdiff_t;   // Is it really needed?
        //using value_type        = TupleRow;
        using pointer           = TupleRow*;  // or also value_type*
        //using reference         = K*;  // or also value_type&
        //using reference         = std::unique_ptr<K>;  // or also value_type&
        using reference         = K&;   // or also value_type&

        // Constructor
        keysIterator(void) : m_ptr(nullptr) {}
        //keysIterator(void) : lastKeyClass(std::make_shared<K>()) {}
        //keysIterator(pointer ptr) : m_ptr(ptr) {}
        //keysIterator(IStorage *my_instance, Prefetch *my_P) : instance(my_instance), P(my_P) {m_ptr = P->get_cnext();}
        keysIterator(StorageDict *my_instance, Prefetch *my_P) {
		P = my_P;
		instance = my_instance;
		m_ptr = P->get_cnext();
		DBG(" m_ptr == " << (uint64_t)m_ptr);
		DBG( " PAYLOAD == "<<(int64_t)m_ptr->get_payload());
		}

        // Operators
        reference operator*() const {
            char *valueToReturn=nullptr;
            DBG(" m_ptr == " << (uint64_t)m_ptr);
            DBG(" BEFORE Get Payload...");
            char *valueFromQuery = (char *)(m_ptr->get_payload());
            DBG(" BEFORE Extracting values...");
	        char *keyBuffer;
	        // if the key only have one attribute, valueToReturn contains that attribute
	        // if the key has several attributes, valueToReturn is a pointer to the memory containing the attributes
	        if ((instance->partitionKeys.size() + instance->clusteringKeys.size()) == 1) {
	            std::pair<uint16_t, uint16_t> keySize = instance->getDataWriter()->get_metadata()->get_keys_size();
	            uint64_t partKeySize = keySize.first;
	            uint64_t clustKeySize = keySize.second;
                valueToReturn = (char *) malloc (partKeySize+clustKeySize);
                instance->extractMultiValuesFromQueryResult(valueFromQuery, valueToReturn, KEYS);
		        keyBuffer = valueToReturn;
            } else { // more than one attribute
                instance->extractMultiValuesFromQueryResult(valueFromQuery, &valueToReturn, KEYS);
		        keyBuffer = (char *) &valueToReturn;
            }
	        K* lastKeyClass = new  K(instance, keyBuffer); //instance a new KeyClass to be intitialize with the values in the buffer: case multiattribute
            return *lastKeyClass;
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
        StorageDict *instance;
    };

/***
    d = //dictionary [int, float]:[values]
    for (keysIterator s = d.begin(); s != d.end(); s ++) {
        (*s)        <== buffer con int+float (*m_ptr)
        (s)         <== Iterador
        (s->xxx)    <== NOT SUPPORTED
    }
***/
keysIterator begin() {
        // Create thread and ask Casandra for data

        config_map iterator_config = getCurrentSession()->config;
        iterator_config["type"]="keys"; // Request a prefetcher for 'keys' only
        return keysIterator(this,
		 getCurrentSession()->getStorageInterface()->get_iterator(getDataAccess()->get_metadata(), iterator_config));
}

keysIterator end()   { return keysIterator(); } // NULL is the placeholder for last element

private:
    //Istorage * sd;
    std::map<K,V> sd;
    std::vector<std::pair<std::string, std::string>> partitionKeys;
    std::vector<std::pair<std::string, std::string>> clusteringKeys;
    std::vector<std::pair<std::string, std::string>> valuesDesc;
    
};



#endif
