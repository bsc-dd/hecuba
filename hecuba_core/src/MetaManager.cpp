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


void MetaManager::register_obj(const uint64_t *storage_id, const std::string &name,
        const ArrayMetadata *numpy_meta) const {
    
    
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
    size += sizeof(uint32_t) * numpy_meta->dims.size();

    //plus the other metas
    size += sizeof(numpy_meta->elem_size) + sizeof(numpy_meta->inner_type) + sizeof(numpy_meta->partition_type);
    
    //allocate plus the bytes counter
    unsigned char *byte_array = (unsigned char *) malloc(size+ sizeof(uint64_t));
    unsigned char *name_array = (unsigned char *) malloc(size_name);
 
    void *values = (char *) malloc(sizeof(char *)*2);

    // copy table name
    memcpy(name_array, c_name, strlen(c_name)+1); //lgarrobe
    //int offset_values =strlen(c_name)+1;
    
    // Copy num bytes
    memcpy(byte_array+offset, &size, sizeof(uint64_t));
    offset += sizeof(uint64_t);


    //copy everything from the metas
    memcpy(byte_array + offset, &numpy_meta->elem_size, sizeof(numpy_meta->elem_size));
    offset += sizeof(numpy_meta->elem_size); 

    memcpy(byte_array + offset, &numpy_meta->inner_type, sizeof(numpy_meta->inner_type));
    offset += sizeof(numpy_meta->inner_type);
    memcpy(byte_array + offset, &numpy_meta->partition_type, sizeof(numpy_meta->partition_type));
    offset += sizeof(numpy_meta->partition_type);
    memcpy(byte_array + offset, numpy_meta->dims.data(), sizeof(uint32_t) * numpy_meta->dims.size());
    offset +=sizeof(numpy_meta->inner_type);



    memcpy(values, &name_array, sizeof(unsigned char *));
    int offset_values = sizeof(unsigned char *);
    memcpy(values+offset_values, &byte_array,  sizeof(unsigned char *));


    try{
       this->writer->write_to_cassandra(keys, values);
       
    } 
    catch (std::exception &e) {
        std::cerr << "Error writing in registering" <<std::endl;
        std::cerr << e.what();
        throw e;
    }

}
