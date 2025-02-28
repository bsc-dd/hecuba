#ifndef _STORAGENUMPY_
#define _STORAGENUMPY_

#include <map>
#include <iostream>
#include <type_traits>
#include "ObjSpec.h"
#include "debug.h"
#include "IStorage.h"
#include "UUID.h"
#include "ArrayDataStore.h"
#include "HecubaExtrae.h"


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

class StorageNumpy:virtual public IStorage {
    public:

        void *data = nullptr;   /* Pointer to memory containing the object. */
        std::vector<uint32_t> metas;
        ArrayMetadata numpy_metas; /* Pointer to memory containing the metadata. READ ONLY. DO NOT FREE. This object does NOT own the memory! */
        void initObjSpec(char dtype) {
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
            numpy_metas.typekind = dtype;
            // 
            // yolandab: delay the initialization of the class name outside the constructor 
            // to get the actual name of the class. m
            //initializeClassName("StorageNumpy");
        }

        StorageNumpy() {
            HecubaExtrae_event(HECUBAEV, HECUBA_SN|HECUBA_INSTANTIATION);
            initObjSpec('f'); // Float by default
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        /* Note: 'dtype' uses the array-interface api which is slightly different from Python
         *  array-interface:https://numpy.org/doc/1.21/reference/arrays.interface.html#arrays-interface
         *  python: https://numpy.org/doc/stable/reference/arrays.dtypes.html */
        StorageNumpy(void *datasrc, const std::vector<uint32_t> &metas,char dtype='f') { // TODO: change 'dtype' type to string or better add another parameter with the size
            HecubaExtrae_event(HECUBAEV, HECUBA_SN|HECUBA_INSTANTIATION);
            setNumpy(datasrc, metas, dtype);
            initObjSpec(dtype);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        void setNumpy(void *datasrc, const std::vector<uint32_t>&metas, char dtype='f') {
            // Transform user metas to ArrayMetadata
            this->metas = metas; // make a copy of user 'metas'
            uint64_t numpy_size = extractNumpyMetaData(metas, dtype, this->numpy_metas );

            // Make a copy of 'datasrc'
            if (this->data != nullptr)
                free(this->data);
            this->data = malloc(numpy_size);
            memcpy(this->data, datasrc, numpy_size);
        }

        // yolandab: setNumpy to be used after a getByAlias to support stream in of a numpy
        // In the case of StorageDict the getByAlias method do not populate the dictionary, we delay the access to cassandra until the moment when the user access the elements
        // In the case of StorageNuympy we have not define an interface to access the StorageNumpy so we decided to include in the getByAlias the retrieval of the whole StorageNumpy
        // If we allow stream-in of StorageNumpy we need to allow the instantiation of a persistent storageNumpy without reading the data so I have moved the read from the getByAlias 
        // (method setPersistence) to the new setNumpy without parameters. If the numpy is not a StorageStream then we use the setNumpy method in the setPersistence 
        // (the behaviour is as before). If the numpy is a StorageStream that already exists and the user wants to stream out the data, then the user needs to call the 
        // setNumpy method in the code.
        // TODO: define an interface to access elements in a StorageNumpy
        //

        void setNumpy() {
            // check that we have executed the setPersistence method
            DBG("StorageNumpy:setNumpy: uuid: " << UUID::UUID2str(getStorageID()));
            if (arrayStore == nullptr) {
                throw ModuleException("StorageNumpy::setNumpy  setNumpy without parameters can only be used on a persistent object initialized from storage (get_by_alias method)"); 
            } 
            if (this->data != nullptr)
                free(this->data);
            this->data = malloc(numpy_metas.get_array_size());
            std::list<std::vector<uint32_t>> coord = {};
            DBG("StorageNumpy:setNumpy: launching read from cassandra ");
            arrayStore->read_numpy_from_cas_by_coords(getStorageID(), numpy_metas, coord, data);
        }

        // StorageNumpy sn = misn;
        StorageNumpy(const StorageNumpy &src) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SN|HECUBA_INSTANTIATION);
            //JJ StorageNumpy(src.data, src.metas);
            // Transform user metas to ArrayMetadata
            this->metas = src.metas; // make a copy of user 'metas'
            uint64_t numpy_size = extractNumpyMetaData(src.metas, src.numpy_metas.typekind, this->numpy_metas);

            // Make a copy of 'datasrc'
            this->data = malloc(numpy_size);
            memcpy(this->data, src.data, numpy_size);

	    this->arrayStore = src.arrayStore;

            initObjSpec(src.numpy_metas.typekind);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        // StorageNumpy sn; sn = misn;
        StorageNumpy &operator = (const StorageNumpy & w) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SN|HECUBA_ASSIGNMENT);
            this->metas = w.metas;
            this->numpy_metas=w.numpy_metas;
            uint64_t numpy_size = extractNumpyMetaData(metas, numpy_metas.typekind, this->numpy_metas);
            this->data = malloc(numpy_size);
            memcpy(this->data, w.data, numpy_size);
	    this->arrayStore = w.arrayStore;
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
            return *this;
        }

        ~StorageNumpy() {

            HecubaExtrae_event(HECUBAEV, HECUBA_SN|HECUBA_DESTROY);
            DBG( " StorageNumpy::Destructor " << UUID::UUID2str(getStorageID())<<std::endl);
            if (this->data != nullptr) {
                free (this->data);
            }
            getCurrentSession().unregisterObject(arrayStore);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        void generatePythonSpec() {
            std::string StreamPart="";
            if (isStream() ){
                StreamPart=std::string(", StorageStream");
            }
            std::string pythonSpec = PythonDisclaimerString + "from hecuba import StorageNumpy"
                + StreamPart
                + "\n\nclass "
                + getClassName() + "(StorageNumpy"
                + StreamPart
                +"):\n"
                + "   '''\n   '''\n" ;
            setPythonSpec(pythonSpec);
        }

        void assignTableName(const std::string& id_obj, const std::string& id_model) {
            size_t pos= id_obj.find_first_of(".");
            if (getCurrentSession().config["hecuba_sn_single_table"] == "false") {
                this->setTableName(id_obj.substr(pos+1,id_obj.size())); //in the case of StorageObject this will be the name of the class
            } else {
                this->setTableName("HECUBA_StorageNumpy");
                // TODO: Add a check in StorageObj to avoid this name as the table name.
            }
        }

        void initialize_dataAcces() {
            // StorageNumpy
            DBG("StorageNumpy:initialize_dataAccess");
            this->arrayStore = std::make_shared<ArrayDataStore> (getTableName().c_str(),
                    getCurrentSession().config["execution_name"].c_str(),
                    getCurrentSession().getStorageInterface(), getCurrentSession().config);

            getCurrentSession().registerObject(arrayStore,getClassName());
        }

        void enableStreamConsumer(std::string topic) {
            DBG ("StorageNumpy::enableStream Consumer");
            this->arrayStore->getReadCache()->enable_stream((std::map<std::string, std::string>&)getCurrentSession().config);
            // yolandab: enable stream to poll this could be done with the first poll like in python
            this->arrayStore->getReadCache()->enable_stream_consumer(topic.c_str());
        }

        //TODO delete this specialization of getDataWrite
        Writer * getDataWriter() const {
            DBG ("StorageNumpy::getDataWriter");
            DBG ("StorageNumpy::getDataWriter arrayStore " << this->arrayStore);
            DBG ("StorageNumpy::getDataWriter writeCache " << this->arrayStore->getWriteCache());
            DBG ("StorageNumpy::getDataWriter writer     " << this->arrayStore->getWriteCache()->get_writer());
            return this->arrayStore->getWriteCache()->get_writer();
        }

        void persist_metadata(uint64_t* c_uuid) {
            // Register in hecuba.istorage
            DBG ("StorageNumpy::persist_metadata");
            registerNumpy(this->numpy_metas,  getObjectName(), c_uuid);
        }

        void persist_data() {
            // Dump data into Cassandra
            DBG ("StorageNumpy::persist_data");
            arrayStore->store_numpy_into_cas(getStorageID(), this->numpy_metas, this->data);

        }

        /* setPersistence - Inicializes current instance to conform to uuid object. To be used on an empty instance. */
        void setPersistence (uint64_t *uuid) {
            // FQid_model: Fully Qualified name for the id_model: module_name.id_model
            DBG("StorageNumpy: setPersistence: UUID"<<UUID::UUID2str(uuid)); 
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

            numpy_metas = row.numpy_metas;
            this->metas = numpy_metas.dims;

            init_persistent_attributes(tablename, uuid);
            // Create READ/WRITE cache accesses
            initialize_dataAcces();

            //yolandab: move these 3 lines to  read numpy  to setNumpy
            //this->data = malloc(numpy_metas.get_array_size());
            //std::list<std::vector<uint32_t>> coord = {};
            //arrayStore->read_numpy_from_cas_by_coords(uuid, numpy_metas, coord, data);
            //if the StorageNumpy is not defined to be a stream we load the data. If it is a numpy to stream out the user should call setNumpy in the code
            if (!isStream()) {
                setNumpy();
            }
        }

        void send(void) {
            DBG("DEBUG: IStorage::send: sending numpy. Size "<< numpy_metas.get_array_size());

            uint64_t total_size = numpy_metas.get_array_size();
            uint64_t offset = 0;
            uint64_t partition_size = 262144; //arbitrarily use a large power of two number: 2^18
            char* tmp = (char*)data;
            uint64_t pending = total_size-offset;
            while (pending > 0) {
                uint64_t actual_size;
                //check if last partition is less than partition_size
                if (pending < partition_size) {
                    actual_size = pending;
                } else {
                    actual_size = partition_size;
                }
                getDataWriter()->send_event(UUID::UUID2str(getStorageID()).c_str(), &tmp[offset], actual_size);

                offset += actual_size;
                pending = total_size-offset;
            }
        }

        void poll(void) {
            DBG("StorageNumpy: poll");
            if (this->data)
                free(this->data);
            this->data = malloc(numpy_metas.get_array_size());
            this->arrayStore->getReadCache()->poll(UUID::UUID2str(getStorageID()).c_str(), (char*)data, numpy_metas.get_array_size());
        }

        void writePythonSpec() {} // StorageNumpy do not have python specification

    private:

        std::shared_ptr<ArrayDataStore> arrayStore = nullptr; /* Cache of written/read elements */

        uint32_t getDtypeSize(char dtype) const {
            switch(dtype) {
                case 'f': //NPY_FLOAT
                          return sizeof(double);
                case 'b': //NPY_BOOL
                          return sizeof(char);
                case 'i': //NPY_INT
                case 'u': //NPY_UINT
                          return sizeof(int64_t);
                default: {
                             std::string msg ("StorageNumpy::getDtypeSize: Unsupported type [");
                             msg += dtype;
                             msg += "] ";
                             throw ModuleException(msg);
                         }
            }
        }

        uint64_t extractNumpyMetaData(const std::vector<uint32_t> &raw_numpy_meta, char dtype, ArrayMetadata &arr_metas) {
            std::vector <uint32_t> dims;
            std::vector <uint32_t> strides;

            arr_metas.elem_size = getDtypeSize(dtype); // TODO: This should be a parameter!
            DBG("StorageNumpy::extractNumpyMetadata. dtype "<< dtype);
            DBG("StorageNumpy::extractNumpyMetadata. elem_size "<< arr_metas.elem_size);
            // decode void *metadatas
            uint64_t acum=1;
            uint64_t numpy_size=0;
            for (uint32_t i=0; i < raw_numpy_meta.size(); i++) {
                dims.push_back( raw_numpy_meta[i]);
                acum *= raw_numpy_meta[i];
            }
            numpy_size = acum;
            for (uint32_t i=0; i < raw_numpy_meta.size(); i++) {
                acum/=raw_numpy_meta[i];
                strides.push_back(acum * arr_metas.elem_size);
            }
            uint32_t flags=NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED;

            arr_metas.dims = dims;
            arr_metas.strides = strides;
            arr_metas.flags = flags;
            arr_metas.partition_type = ZORDER_ALGORITHM;
            arr_metas.typekind = dtype;
            arr_metas.byteorder = '=';
            return numpy_size*arr_metas.elem_size;
        }

        void registerNumpy(ArrayMetadata &numpy_meta, std::string name, uint64_t* uuid) {

            //std::cout<< "DEBUG: HecubaSession::registerNumpy BEGIN "<< name << UUID::UUID2str(uuid)<<std::endl;
            DBG("StorageNumpy::registerNumpy: name: " << name <<" UUID " << UUID::UUID2str(uuid));
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

            DBG("StorageNumpy::registerNumpy. numpy_meta elem_size: "<< numpy_meta.elem_size);
            DBG("StorageNumpy::registerNumpy. numpy_meta partition_type: "<<numpy_meta.partition_type);
            DBG("StorageNumpy::registerNumpy. numpy_meta flags: "<< numpy_meta.flags);
            DBG("StorageNumpy::registerNumpy. numpy_meta typekind: "<< numpy_meta.typekind);
            DBG("StorageNumpy::registerNumpy. numpy_meta byteorder: "<< numpy_meta.byteorder);

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

            // yolandab: deal with classes that inherit from StorageNumpy. Needed to deal with StorageStream
            //char *class_name=(char*)malloc(strlen("hecuba.hnumpy.StorageNumpy")+1);
            //strcpy(class_name, "hecuba.hnumpy.StorageNumpy");
            DBG("StorageNumpy::registerNumpy before getClassName");
            const char*  class_nameSRC = getIdModel().c_str();
            char *class_name=(char*)malloc(strlen(class_nameSRC)+1);
            strcpy(class_name, class_nameSRC);
            std::cout<< "DEBUG: HecubaSession::registerNumpy &class_name = "<<class_name<<std::endl;
            memcpy(values+offset_values, &class_name, sizeof(unsigned char *)); //class_name
            DBG("StorageNumpy::registerNumpy after copying class_name");
            offset_values += sizeof(unsigned char *);

            //std::cout<< "DEBUG: HecubaSession::registerNumpy &name = "<<name_array<<std::endl;
            memcpy(values+offset_values, &name_array, sizeof(unsigned char *)); //name
            offset_values += sizeof(unsigned char *);

            //std::cout<< "DEBUG: HecubaSession::registerNumpy &np_meta = "<<byte_array<<std::endl;
            memcpy(values+offset_values, &byte_array,  sizeof(unsigned char *)); // numpy_meta
            offset_values += sizeof(unsigned char *);


            try {
                //getCurrentSession().getNumpyMetaWriter()->write_to_cassandra(keys, values);
                //getCurrentSession().getNumpyMetaWriter()->wait_writes_completion(); // Ensure hecuba.istorage get all updates SYNCHRONOUSLY (to avoid race conditions with poll that may request a build_remotely on this new object)!
                arrayStore->getMetaDataCache()->get_writer()->write_to_cassandra(keys, values);
                // JJ TODO arrayStore->getMetaDataCache()->get_writer()->wait_writes_completion(); // Ensure hecuba.istorage get all updates SYNCHRONOUSLY (to avoid race conditions with poll that may request a build_remotely on this new object)!
                arrayStore->getMetaDataCache()->get_writer()->wait_writes_completion(); // Ensure hecuba.istorage get all updates SYNCHRONOUSLY (to avoid race conditions with poll that may request a build_remotely on this new object)!
            }
            catch (std::exception &e) {
                std::cerr << "HecubaSession::registerNumpy: Error writing" <<std::endl;
                std::cerr << e.what();
                throw e;
            };

            DBG("StorageNumpy::registerNumpy. hecuba.istorage updated");
        }

        bool is_create_table_required(void) {
            // If a single table is used, then it is NOT required to create a
            // table at each instantiation (just the first time, which is done
            // at initialization time))
            static bool current_table_required = (getCurrentSession().config["hecuba_sn_single_table"] == "false");
            return current_table_required;
        }
    
};

#endif /* _STORAGENUMPY_ */
