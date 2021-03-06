#include "MetaManager.h"


MetaManager::MetaManager(const TableMetadata *table_meta, CassSession *session,
                         std::map<std::string, std::string> &config) {
    

    this->writer = new Writer(table_meta, session, config);

}

MetaManager::~MetaManager() {
    delete (this->writer);
}

void MetaManager::create_data_model(const std::string &create_table_query) const {
    /* TODO
    Distributed agreement on table creation:
    1) r = insert if not exists on agreement table
    2) if !r -> create table
    3) else wait*/

    throw ModuleException("Not implemented yet");
}

/* name Full name of the object to register 'keyspace.table_name' */
void MetaManager::register_obj(const uint64_t *storage_id, const std::string &name,
        const ArrayMetadata &numpy_meta) const {
    
    
    void *keys = std::malloc(sizeof(uint64_t *));
    uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);//new uint64_t[2];
    c_uuid[0] = *storage_id;
    c_uuid[1] = *(storage_id + 1);


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
		+ sizeof(uint32_t)*numpy_meta.strides.size()
		+ sizeof(numpy_meta.typekind)
		+ sizeof(numpy_meta.byteorder);
    
    //allocate plus the bytes counter
    unsigned char *byte_array = (unsigned char *) malloc(size+ sizeof(uint64_t));
    unsigned char *name_array = (unsigned char *) malloc(size_name);
 

    // copy table name
    memcpy(name_array, c_name, size_name); //lgarrobe
    //int offset_values =strlen(c_name)+1;
    
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
    void *values = (char *) malloc(sizeof(char *)*4);

    uint64_t *base_numpy = (uint64_t *) malloc(sizeof(uint64_t) * 2);//new uint64_t[2];
    memcpy(base_numpy, c_uuid, sizeof(uint64_t)*2);
    std::memcpy(values, &base_numpy, sizeof(uint64_t *));  // base_numpy
    offset_values += sizeof(unsigned char *);

    char *class_name=(char*)malloc(strlen("hecuba.hnumpy.StorageNumpy")+1);
    strcpy(class_name, "hecuba.hnumpy.StorageNumpy");
    memcpy(values+offset_values, &class_name, sizeof(unsigned char *));
    offset_values += sizeof(unsigned char *);

    memcpy(values+offset_values, &name_array, sizeof(unsigned char *));
    offset_values += sizeof(unsigned char *);

    memcpy(values+offset_values, &byte_array,  sizeof(unsigned char *));
    offset_values += sizeof(unsigned char *);

    try{
       this->writer->write_to_cassandra(keys, values);
       
    } 
    catch (std::exception &e) {
        std::cerr << "Error writing in registering" <<std::endl;
        std::cerr << e.what();
        throw e;
    }

}
