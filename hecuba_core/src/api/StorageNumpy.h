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
            initializeClassName("StorageNumpy");
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

        void setNumpy(void *datasrc, const std::vector<uint32_t>&metas, char dtype) {
            // Transform user metas to ArrayMetadata
            this->metas = metas; // make a copy of user 'metas'
            uint32_t numpy_size = extractNumpyMetaData(metas, dtype, this->numpy_metas );

            // Make a copy of 'datasrc'
            if (this->data != nullptr)
                free(this->data);
            this->data = malloc(numpy_size);
            memcpy(this->data, datasrc, numpy_size);
        }

        // StorageNumpy sn = misn;
        StorageNumpy(const StorageNumpy &src) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SN|HECUBA_INSTANTIATION);
            //JJ StorageNumpy(src.data, src.metas);
            // Transform user metas to ArrayMetadata
            this->metas = src.metas; // make a copy of user 'metas'
            uint32_t numpy_size = extractNumpyMetaData(src.metas, src.numpy_metas.typekind, this->numpy_metas);

            // Make a copy of 'datasrc'
            this->data = malloc(numpy_size);
            memcpy(this->data, src.data, numpy_size);

            initObjSpec(src.numpy_metas.typekind);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
        }

        // StorageNumpy sn; sn = misn;
        StorageNumpy &operator = (const StorageNumpy & w) {
            HecubaExtrae_event(HECUBAEV, HECUBA_SN|HECUBA_ASSIGNMENT);
            this->metas = w.metas;
            this->numpy_metas=w.numpy_metas;
            uint32_t numpy_size = extractNumpyMetaData(metas, numpy_metas.typekind, this->numpy_metas);
            this->data = malloc(numpy_size);
            memcpy(this->data, w.data, numpy_size);
            HecubaExtrae_event(HECUBAEV, HECUBA_END);
            return *this;
        }

        ~StorageNumpy() {

            HecubaExtrae_event(HECUBAEV, HECUBA_SN|HECUBA_DESTROY);
            //std::cout << " StorageNumpy::Destructor " << UUID::UUID2str(getStorageID())<<std::endl;
            if (this->data != nullptr) {
                free (this->data);
            }
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
            this->arrayStore = std::make_shared<ArrayDataStore> (getTableName().c_str(),
                    getCurrentSession().config["execution_name"].c_str(),
                    getCurrentSession().getStorageInterface(), getCurrentSession().config);

            getCurrentSession().registerObject(arrayStore,getClassName());
        }

        Writer * getDataWriter() const {
            std::cout<< "getDataWriter numpy" << std::endl;
            return this->arrayStore->getWriteCache()->get_writer();
        }

        void persist_metadata(uint64_t* c_uuid) {
            // Register in hecuba.istorage
            registerNumpy(this->numpy_metas,  getObjectName(), c_uuid);
        }

        void persist_data() {
            // Dump data into Cassandra
            arrayStore->store_numpy_into_cas(getStorageID(), this->numpy_metas, this->data);

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

            numpy_metas = row.numpy_metas;
            this->metas = numpy_metas.dims;

            init_persistent_attributes(tablename, uuid);
            // Create READ/WRITE cache accesses
            initialize_dataAcces();

            this->data = malloc(numpy_metas.get_array_size());
            std::list<std::vector<uint32_t>> coord = {};
            arrayStore->read_numpy_from_cas_by_coords(uuid, numpy_metas, coord, data);
        }

        void send(void) {
            DBG("DEBUG: IStorage::send: sending numpy. Size "<< numpy_metas.get_array_size());
            getDataWriter()->send_event((char *) data, numpy_metas.get_array_size());
        }

        void writePythonSpec() {} // StorageNumpy do not have python specification

    private:

        std::shared_ptr<ArrayDataStore> arrayStore = nullptr; /* Cache of written/read elements */

        uint32_t getDtypeSize(char dtype) const {
            switch(dtype) {
                case 'f': //NPY_FLOAT
                          return sizeof(double);
                case 'i': //NPY_BYTE
                case 'u': //NPY_UBYTE
                case 'b': //NPY_BOOL
                          return sizeof(char);
                default: {
                             std::string msg ("StorageNumpy::getDtypeSize: Unsupported type [");
                             msg += dtype;
                             msg += "] ";
                             throw ModuleException(msg);
                         }
            }
        }

        uint32_t extractNumpyMetaData(const std::vector<uint32_t> &raw_numpy_meta, char dtype, ArrayMetadata &arr_metas) {
            std::vector <uint32_t> dims;
            std::vector <uint32_t> strides;

            arr_metas.elem_size = getDtypeSize(dtype); // TODO: This should be a parameter!
            // decode void *metadatas
            uint32_t acum=1;
            uint32_t numpy_size=0;
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
                //getCurrentSession().getNumpyMetaWriter()->write_to_cassandra(keys, values);
                //getCurrentSession().getNumpyMetaWriter()->wait_writes_completion(); // Ensure hecuba.istorage get all updates SYNCHRONOUSLY (to avoid race conditions with poll that may request a build_remotely on this new object)!
                arrayStore->getMetaDataCache()->get_writer()->write_to_cassandra(keys, values);
                arrayStore->getMetaDataCache()->get_writer()->wait_writes_completion(); // Ensure hecuba.istorage get all updates SYNCHRONOUSLY (to avoid race conditions with poll that may request a build_remotely on this new object)!
            }
            catch (std::exception &e) {
                std::cerr << "HecubaSession::registerNumpy: Error writing" <<std::endl;
                std::cerr << e.what();
                throw e;
            };

        }
    
};

#endif /* _STORAGENUMPY_ */
