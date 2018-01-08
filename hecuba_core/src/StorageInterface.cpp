#include "StorageInterface.h"


StorageInterface::StorageInterface(int nodePort, std::string contact_points) {
    cluster = new ClusterConfig(nodePort, contact_points);
}

StorageInterface::~StorageInterface() {
    delete cluster;
}


CacheTable *StorageInterface::make_cache(const char *table, const char *keyspace,
                                         std::vector<std::map<std::string, std::string>> &keys_names,
                                         std::vector<std::map<std::string, std::string>> &columns_names,
                                         std::map<std::string, std::string> &config) {
    if (!cluster) throw ModuleException("StorageInterface not connected to any node");
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, cluster->get_session());
    return new CacheTable(table_meta, cluster->get_session(), config);
}


Writer *StorageInterface::make_writer(const char *table, const char *keyspace,
                                      std::vector<std::map<std::string, std::string>> &keys_names,
                                      std::vector<std::map<std::string, std::string>> &columns_names,
                                      std::map<std::string, std::string> &config) {
    if (!cluster) throw ModuleException("StorageInterface not connected to any node");
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, cluster->get_session());
    return new Writer(table_meta, cluster->get_session(), config);

}


Writer *StorageInterface::make_writer(const TableMetadata *table_meta,
                                      std::map<std::string, std::string> &config) {
    if (!cluster) throw ModuleException("StorageInterface not connected to any node");
    return new Writer(table_meta, cluster->get_session(), config);

}

/*** ITERATOR METHODS AND SETUP ***/

/***
 * This one retrives the keys comprised on its ranges and the columns if any, assuming partid is keys_names[0]
 * @param table
 * @param keyspace
 * @param keys_names
 * @param columns_names
 * @param tokens
 * @param prefetch_size
 * @return
 */
Prefetch *StorageInterface::get_iterator(const char *table, const char *keyspace,
                                         std::vector<std::map<std::string, std::string>> &keys_names,
                                         std::vector<std::map<std::string, std::string>> &columns_names,
                                         const std::vector<std::pair<int64_t, int64_t>> &tokens,
                                         std::map<std::string, std::string> &config) {
    if (!cluster) throw ModuleException("StorageInterface not connected to any node");
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, cluster->get_session());
    return new Prefetch(tokens, table_meta, cluster->get_session(), config);
}


Prefetch *StorageInterface::get_iterator(const TableMetadata *table_meta,
                                         const std::vector<std::pair<int64_t, int64_t>> &tokens,
                                         std::map<std::string, std::string> &config) {
    if (!cluster) throw ModuleException("StorageInterface not connected to any node");
    return new Prefetch(tokens, table_meta, cluster->get_session(), config);
}