#include "StorageInterface.h"

#define default_io_threads 2
#define default_low_watermark 20000
#define default_high_watermark 17000000
#define bytes_high_watermark 17000000 //>128elements^3D * 8B_Double


StorageInterface::StorageInterface(int nodePort, std::string contact_points) {
    CassFuture *connect_future = NULL;
    cluster = cass_cluster_new();
    session = cass_session_new();

    // add contact points
    if (contact_points.empty()) contact_points = "127.0.0.1";
    cass_cluster_set_contact_points(cluster, contact_points.c_str());
    cass_cluster_set_port(cluster, nodePort);
    cass_cluster_set_token_aware_routing(cluster, cass_true);


    char *env_path = std::getenv("WRITE_IO_THREADS");
    if (env_path != nullptr) cass_cluster_set_num_threads_io(cluster, (int32_t) std::atoi(env_path));
    else cass_cluster_set_num_threads_io(cluster, default_io_threads);

    env_path = std::getenv("WRITE_LOW_WATERMARK");
    if (env_path != nullptr) cass_cluster_set_pending_requests_low_water_mark(cluster, (int32_t) std::atoi(env_path));
    else cass_cluster_set_pending_requests_low_water_mark(cluster, (int32_t) default_low_watermark);

    env_path = std::getenv("WRITE_HIGH_WATERMARK");
    if (env_path != nullptr) cass_cluster_set_pending_requests_high_water_mark(cluster, (int32_t) std::atoi(env_path));
    else cass_cluster_set_pending_requests_high_water_mark(cluster, (int32_t) default_high_watermark);


    cass_cluster_set_write_bytes_high_water_mark(cluster, bytes_high_watermark); //>128elements^3D * 8B_Double


    //unsigned int uiRequestTimeoutInMS = 30000;
    //cass_cluster_set_core_connections_per_host (cluster, 4);
    //cass_cluster_set_request_timeout (cluster, uiRequestTimeoutInMS);


    // Provide the cluster object as configuration to connect the session
    connect_future = cass_session_connect(session, cluster);
    CassError rc = cass_future_error_code(connect_future);
    if (rc != CASS_OK) {
        std::string msg(cass_error_desc(rc));
        const char *dmsg;
        size_t l;
        cass_future_error_message(connect_future, &dmsg, &l);
        std::string msg2(dmsg, l);
        throw ModuleException(msg + " - " + msg2);
    }
    cass_future_free(connect_future);
}


StorageInterface::~StorageInterface() {
    disconnectCassandra();
}


int StorageInterface::disconnectCassandra() {
    if (session != NULL) {
        CassFuture *close_future = cass_session_close(session);
        cass_future_free(close_future);
        cass_session_free(session);
        cass_cluster_free(cluster);
        session = NULL;
    }
    return 0;
}


CacheTable *StorageInterface::make_cache(const char *table, const char *keyspace,
                                         std::vector<std::map<std::string, std::string>> &keys_names,
                                         std::vector<std::map<std::string, std::string>> &columns_names,
                                         std::map<std::string, std::string> &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
    return new CacheTable(table_meta, session, config);
}


Writer *StorageInterface::make_writer(const char *table, const char *keyspace,
                                      std::vector<std::map<std::string, std::string>> &keys_names,
                                      std::vector<std::map<std::string, std::string>> &columns_names,
                                      std::map<std::string, std::string> &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
    return new Writer(table_meta, session, config);

}


Writer *StorageInterface::make_writer(const TableMetadata *table_meta,
                                      std::map<std::string, std::string> &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    return new Writer(table_meta, session, config);

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
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
    return new Prefetch(tokens, table_meta, session, config);
}


Prefetch *StorageInterface::get_iterator(const TableMetadata *table_meta,
                                         const std::vector<std::pair<int64_t, int64_t>> &tokens,
                                         std::map<std::string, std::string> &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    return new Prefetch(tokens, table_meta, session, config);
}