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

    void *keys = std::malloc(sizeof(storage_id));
    std::memcpy(keys, &storage_id, sizeof(uint64_t *));

    //char *c_name = new char[name.length() + 1];
    char *c_name = (char *) std::malloc(name.length() + 1);
    std::memcpy(c_name, name.c_str(), name.length() + 1);

    void *values = std::malloc(sizeof(char *) + sizeof(numpy_meta));
    std::memcpy(values, &c_name, sizeof(char *));

    //std::memcpy(values+sizeof(c_name), numpy_meta, sizeof(numpy_meta));
    this->writer->write_to_cassandra(keys, values);
}
