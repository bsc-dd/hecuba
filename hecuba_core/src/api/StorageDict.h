#ifndef _STORAGEDICT_
#define _STORAGEDICT_

#include <map>
#include <iostream>
#include <type_traits>
#include "ObjSpec.h"
#include "debug.h"
#include "IStorage.h"
#include "KeyClass.h"
#include "ValueClass.h"
#include "UUID.h"
#include "StorageNumpy.h"
#include "HecubaExtrae.h"

// C should be the class defined by the user
// HecubaSession s; --> should be called just once in the user code and then shoul be accessible from all IStorages. How?
// the new version of registerObject can be implemented in IStorage and receives as parametr the class_name that we extract from the template.
// The registerObjet needs acccess to the session to store it in currentSession and to keep the object in the alive_objects list. The dataModel
// can be delete (I think) because it is only used once in StorageObject (getAttr) and can be replaced by the getObjSpec of the IStorage.

template<class K, class V, class C>

class StorageDict:virtual public IStorage {

#define ISKEY true

    public:
        void initObjSpec() {
            K key;
            V v;
            partitionKeys=key.getPartitionKeys();
            clusteringKeys=key.getClusteringKeys();
            valuesDesc=v.getValuesDesc("val_");
            ObjSpec dictSpec;
            dictSpec=ObjSpec(ObjSpec::valid_types::STORAGEDICT_TYPE, partitionKeys, clusteringKeys, valuesDesc,"");
            setObjSpec(dictSpec);
            // extract class name from C and call the new registerObject. And the session??? HecubaSession::get() ?
            //
            int32_t status;
            std::string class_name = abi::__cxa_demangle(typeid(C).name(),NULL,NULL,&status);
            initializeClassName (class_name); // implemented in IStorage.cpp same code that current registerClassName of HecubaSession
        }

        StorageDict() {
            HecubaExtrae_event(HECUBAEV, HECUBA_SD|HECUBA_INSTANTIATION);
            //std::cout << "StorageDict:: default constructor this "<< this <<std::endl;
            initObjSpec();
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }
        StorageDict(const StorageDict<K,V,C>& sdsrc) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SD|HECUBA_INSTANTIATION);
            //std::cout << "StorageDict:: copy constructor this "<< this << " from "<< &sdsrc << std::endl;
            *this = sdsrc;
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }


        // c++ only calls implicitly the constructor without parameters. To invoke this constructor we need to add to the user class an explicit call to this
        StorageDict(const std::string& name) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SD|HECUBA_INSTANTIATION);
            initObjSpec();
            id_obj=name;
            pending_to_persist=true;
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        ~StorageDict() {
            HecubaExtrae_event(HECUBAEV, HECUBA_SD|HECUBA_DESTROY);
            getCurrentSession().unregisterObject(getDataAccess());
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }


        // sd[k] = v or v = sd[k] 
        // return a reference to allow sd[k]=v
        // in the operator = for ValueClass we determine in which case we are
        V& operator[](K &key) {
            V *v = new V(static_cast<IStorage*>(this), key.getKeysBuffer(),key.getTotalSize()); //static_cast<IStorage*> is REQUIRED because we need to access later to its getStorageID method
            return *v;
        }

        //copy assignment
        StorageDict<K,V,C> &operator = (const StorageDict<K,V,C> &sdsrc){
            HecubaExtrae_event(HECUBAEV, HECUBA_SD|HECUBA_ASSIGNMENT);
            if (this != &sdsrc) {
                this->IStorage::operator=(sdsrc); //Inherit IStorage attributes
                sd = sdsrc.sd;
                partitionKeys = sdsrc.partitionKeys;
                clusteringKeys = sdsrc.clusteringKeys;
                valuesDesc = sdsrc.valuesDesc;
            }
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
            return *this;
        }


        // It generates the python specification for the class during the registration of the object
        void generatePythonSpec() {
            std::string StreamPart="";
            if (isStream() ){
                StreamPart=std::string(", StorageStream");
            }
            std::string pythonSpec = PythonDisclaimerString +
                "from hecuba import StorageDict"
                + StreamPart +
                + "\n\nclass "
                + getClassName() + "(StorageDict"
                + StreamPart
                + "):\n"
                + "   '''\n   @TypeSpec dict <<";

            std::string itemSpec = "";
            for (std::vector<std::pair<std::string,std::string>>::iterator it=partitionKeys.begin(); it!=partitionKeys.end(); ++it)
                itemSpec+=it->first + ":"+ ObjSpec::cass_to_hecuba(it->second) + ",";
            for (std::vector<std::pair<std::string,std::string>>::iterator it=clusteringKeys.begin(); it!=clusteringKeys.end(); ++it)
                itemSpec+=it->first + ":"+ ObjSpec::cass_to_hecuba(it->second) + ",";

            pythonSpec += itemSpec.substr(0, itemSpec.size()-1) + ">,"; // replace the last , with a >

            itemSpec = "";
            for (std::vector<std::pair<std::string,std::string>>::iterator it=valuesDesc.begin(); it!=valuesDesc.end(); ++it)
                itemSpec+=it->first + ":"+ ObjSpec::cass_to_hecuba(it->second) + ",";
            pythonSpec += itemSpec.substr(0, itemSpec.size()-1) + ">\n   '''\n"; // replace the last , with a >

            setPythonSpec(pythonSpec);
        }

        void assignTableName(const std::string& id_obj, const std::string& id_model) {
            size_t pos = id_obj.find_first_of(".");
            std::string tablename =id_obj.substr(pos+1, id_obj.size());
            this->setTableName( tablename ); //in the case of StorageObject this will be the name of the class
        }

        void enableStreamConsumer(std::string topic) {
            DBG ("StorageDict::enableStream Consumer");
            this->getDataAccess()->enable_stream((std::map<std::string, std::string>&)getCurrentSession().config);
            // yolandab: enable stream to poll this could be done with the first poll like in python
            this->getDataAccess()->enable_stream_consumer(topic.c_str());
        }

        void persist_metadata(uint64_t* c_uuid) {
            ObjSpec oType = getObjSpec(); 
            std::string insquery = 	std::string("INSERT INTO ") +
                std::string("hecuba.istorage") +
                std::string("(storage_id, name, class_name, primary_keys, columns)") +
                std::string("VALUES ") +
                std::string("(") +
                UUID::UUID2str(c_uuid) + std::string(", ") +
                "'" + getCurrentSession().config["execution_name"] + "." + getTableName() + "'" + std::string(", ") +
                "'" + this->getIdModel() + "'" + std::string(", ") +
                oType.getKeysStr() + std::string(", ") +
                oType.getColsStr() +
                std::string(")");
            HecubaExtrae_event(HECUBACASS, HBCASS_SYNCWRITE);
            CassError rc = getCurrentSession().run_query(insquery);
            HecubaExtrae_event(HECUBACASS, HBCASS_END);
            if (rc != CASS_OK) {
                std::string msg = std::string("StorageDict::persist_metadata: Error executing query ") + insquery;
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
                this->getDataWriter()->send_event(UUID::UUID2str(getStorageID()).c_str(), trow_key, trow_values); // stream value (storage_id/value)
                send_values(value); // If value is an IStorage type stream its contents also
            }
#endif
            this->getDataAccess()->put_crow(trow_key, trow_values);
            delete(trow_key);
            delete(trow_values);
        }

        void getItem(const void* key, void *valuetoreturn) {
            const TableMetadata* writerMD = getDataAccess()->get_metadata();
            /* PRE: value arrives already coded as expected: block of memory with pointers to IStorages or basic values*/
            std::pair<uint16_t, uint16_t> keySize = writerMD->get_keys_size();
            int key_size = keySize.first + keySize.second;

            std::shared_ptr<const std::vector<ColumnMeta> > columns = writerMD->get_keys();

            void *keytosend = deep_copy_attribute_buffer(ISKEY, key, key_size, columns->size());

            std::vector<const TupleRow *> result = getDataAccess()->get_crow(keytosend);

            if (result.empty()) throw ModuleException("IStorage::getItem: key not found in object "+ getName());

            char *query_result= (char*)result[0]->get_payload();

            // WARNING: The order of fields in the TableMetadata and in the model may
            // NOT be the same! Traverse the TableMetadata and construct the User
            // buffer with the same order as the ospec. FIXME

            extractMultiValuesFromQueryResult(query_result, _HECUBA_COLUMNS_, valuetoreturn);

            // TODO this works only for dictionaries of one element. We should traverse the whole vector of values
            // TODO delete the vector of tuple rows and the tuple rows
            return;
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
                        StorageNumpy * result = dynamic_cast<StorageNumpy *>(*((IStorage **)(src+offset))); // 'src' MUST be a valid pointer or it will segfault here...
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
            CacheTable *reader = getCurrentSession().getStorageInterface()->make_cache(this->getTableName().c_str(),
                    getCurrentSession().config["execution_name"].c_str(), *keyNamesDict, *colNamesDict, getCurrentSession().config);
            HecubaExtrae_event(HECUBADBG, HECUBA_REGISTER);
            this->setCache(reader);
            HecubaExtrae_event(HECUBADBG, HECUBA_END);

            delete keyNamesDict;
            delete colNamesDict;
            HecubaExtrae_event(HECUBADBG, HECUBA_REGISTER);
            bool new_element=getCurrentSession().registerObject(getDataAccess(),getClassName());
            HecubaExtrae_event(HECUBADBG, HECUBA_END);
            if (new_element){
            HecubaExtrae_event(HECUBADBG, HECUBA_REGISTER);
                writePythonSpec();
            HecubaExtrae_event(HECUBADBG, HECUBA_END);
            }

        }

        std::vector<std::pair<std::string, std::string>> getPartitionKeys(){
            return partitionKeys;
        }
        std::vector<std::pair<std::string, std::string>> getClusteringKeys(){
            return clusteringKeys;
        }


        /* Iterators */


        // std::map c++ class only implements an iterator that returns pairs of
        // key-value, thus to respect the semantic in the case of StorageDict
        // we do the same and iterate on items
        struct itemsIterator{
            using iterator_category = std::input_iterator_tag;
            using difference_type   = std::ptrdiff_t;   // Is it really needed?
            using pointer           = const TupleRow*;  // or also value_type*
            using reference         = std::pair<K,V>&;   // or also value_type&
            // Constructor
            itemsIterator() : m_ptr(nullptr), current(nullptr) {}

            itemsIterator(StorageDict *my_instance, Prefetch *my_P) {
                P = std::shared_ptr<Prefetch>(my_P);
                instance = my_instance;
                instance_uuid = UUID::UUID2str(instance->getStorageID());
                m_ptr = P->get_cnext();
                current = nullptr; // It will be recovered at access
                DBG(" m_ptr == " << (uint64_t)m_ptr);
                DBG( " PAYLOAD == "<<(int64_t)m_ptr->get_payload());
            }

            itemsIterator(const itemsIterator& src) {
                m_ptr = nullptr; //Ask the prefetcher
                current = nullptr; // Avoid the copy, If iterator is accessed, then it will be recovered
                instance = src.instance;
                instance_uuid = src.instance_uuid;
                P = src.P; // WARNING! both iterators will SHARE the prefetcher!! HERE BE DRAGONS! (advancing one iterator will act on the elements obtained in the other)
            }

            // if streaming is set we do not use prefetcher
            itemsIterator(StorageDict *my_instance) {
                instance = my_instance;
                instance_uuid = UUID::UUID2str(instance->getStorageID());
                m_ptr = instance->getDataAccess()->poll(instance_uuid.c_str())[0]; // poll returns a vector, the first element contains a TupleRow with the key and the value got
                if (m_ptr->isNull()) {delete(m_ptr); m_ptr=nullptr;}
                current = nullptr;
            }
            ~itemsIterator() {
                if (m_ptr!=nullptr){
                    delete m_ptr;
                }
                if (current!=nullptr){
                    delete(current);
                }
            }

            // Operators

            // Prefix increment
            itemsIterator& operator++() {
                if (m_ptr != nullptr) {
                    delete(m_ptr);
                    m_ptr=nullptr;
                }
                if (current != nullptr) { // Remove previous instance
                    delete(current);
                    current = nullptr;
                }
                if (instance->isStream()) {
                    m_ptr = instance->getDataAccess()->poll(instance_uuid.c_str())[0]; // poll returns a vector, the first element contains a TupleRow with the key and the value got
                    if (m_ptr->isNull()) m_ptr=nullptr;
                } else {
                    m_ptr = P->get_cnext(); //CUANDO LLEGA AL FINAL ESTO DEVUELVE NULL..... m_ptr->get_payload deberia estar fallando...... y las comparaciones == y != tampoco entiendo que detecten el fin
                }
                DBG(" m_ptr == " << (uint64_t)m_ptr);
                return *this;
            }

            // Postfix increment
            itemsIterator operator++(int) { itemsIterator tmp = *this; ++(*this); return tmp; } // This creates a NEW iterator that is autodestroyed when exits operator in the case 's++;'. To ensure that the PREFETCHER is destroyed just once, we have defined it as a shared pointer

            friend bool operator== (const itemsIterator& a, const itemsIterator& b) {
                return a.m_ptr == b.m_ptr;
            };
            friend bool operator!= (const itemsIterator& a, const itemsIterator& b) {
                return a.m_ptr != b.m_ptr;
            };

            std::pair<K,V>& operator*() {
                if (current != nullptr) return *current; //Already accessed, return current directly
                current = instantiateCurrent(m_ptr);//It is not guaranteed to be dereferenceable (for example by trying to access beyond the end).
                return *current;
            }

            std::pair<K,V>* operator->() {
                if (current != nullptr) return current;
                current = instantiateCurrent(m_ptr);
                return current;
            }

            private:

            pointer m_ptr =nullptr;
            std::pair<K,V>* current;    // Temporal storage for K,V for current element
            std::shared_ptr<Prefetch> P = nullptr;
            StorageDict *instance;
            std::string instance_uuid; // Calculated UUID (avoid calculating it each time)

            /* instantiateCurrent: Given a TupleRow* retrieves its content as a {key, value} pair. MUST BE 'inline'!!! Otherwise the parameter addresses changes!!*/
            inline std::pair<K,V>* instantiateCurrent(pointer m_ptr) {
                if (m_ptr == nullptr) return nullptr;
                // Retrieve TupleRow values...
                //char *valueToReturn1=nullptr;
                //char *valueToReturn2=nullptr;

                char *valueFromQuery = (char *)(m_ptr->get_payload());
                DBG(" BEFORE Extracting values...");
                char *keyBuffer;
                char *valueBuffer;

                instance->extractMultiValuesFromQueryResult(valueFromQuery, _HECUBA_ROWS_, &keyBuffer, &valueBuffer);

                auto tmp = new std::pair<K,V>();
                K* currentKey = new  K(instance, keyBuffer); //instance a new KeyClass to be initialized with the values in the buffer: case multiattribute
                V* currentValue = new  V(instance, valueBuffer); //instance a new ValueClass to be initialized with the values in the buffer: case multiattribute } else {
                tmp->first  = *currentKey;
                tmp->second = *currentValue;
                return tmp;
            }
        };

        /***
          d = //dictionary [int, float]:[values]
          for (keysIterator s = d.begin(); s != d.end(); s ++) {
          (*s)        <== pair with key and value (current)
          (s)         <== Iterador (accessed like s->first and s->second)
          }
         ***/
        itemsIterator begin() {
            if (isStream()) return itemsIterator(this);
            // Create prefetcher thread to ask Casandra for data
            config_map iterator_config = getCurrentSession().config;
            iterator_config["type"]="items"; // Request a prefetcher for 'keys' and values
            return itemsIterator(this,
                getCurrentSession().getStorageInterface()->get_iterator(getDataAccess()->get_metadata(), iterator_config));
        }

        itemsIterator end()   { return itemsIterator(); } // NULL is the placeholder for last element in case of using Prefetch, and a null tuple (all values set to null) in case of streaming


    private:
        //Istorage * sd;
        std::map<K,V> sd;
        std::vector<std::pair<std::string, std::string>> partitionKeys;
        std::vector<std::pair<std::string, std::string>> clusteringKeys;
        std::vector<std::pair<std::string, std::string>> valuesDesc;
    
};



#endif
