#include "Cache.h"



Cache::Cache(int nodePort, std::string contact_points) {
    CassFuture *connect_future = NULL;
    cluster = cass_cluster_new();
    session = cass_session_new();

    // add contact points
    cass_cluster_set_contact_points(cluster, contact_points.c_str());
    cass_cluster_set_port(cluster, nodePort);
    cass_cluster_set_token_aware_routing(cluster, cass_true);



//  unsigned int uiRequestTimeoutInMS = 30000;
    //cass_cluster_set_num_threads_io (cluster, 2);
    //cass_cluster_set_core_connections_per_host (cluster, 4);
    //cass_cluster_set_request_timeout (cluster, uiRequestTimeoutInMS);
    cass_cluster_set_pending_requests_low_water_mark (cluster, 20000);
    cass_cluster_set_pending_requests_high_water_mark(cluster, 17000000);

    cass_cluster_set_write_bytes_high_water_mark(cluster,17000000); //>128elements^3D * 8B_Double

    // Provide the cluster object as configuration to connect the session
    connect_future = cass_session_connect(session, cluster);
    CassError rc = cass_future_error_code(connect_future);
    if (rc != CASS_OK) {
        std::string msg(cass_error_desc(rc));
        const char *dmsg;
        size_t l;
        cass_future_error_message(connect_future, &dmsg, &l);
        std::string msg2(dmsg, l);
        throw ModuleException(msg+" - "+msg2);
    }
    cass_future_free(connect_future);
}


Cache::~Cache() {
    disconnectCassandra();
}



CacheTable* Cache::makeCache(const char *table,const char *keyspace, std::vector < std::string> keys_names,std::vector < std::string > columns_names, std::vector<std::pair<int64_t, int64_t>> token_ranges,std::map<std::string,std::string> &config) {
    try {

        std::string token_range_pred = "WHERE token("+keys_names[0]+")>=? AND token("+keys_names[0]+")<?;";
        return new CacheTable(std::string(table), std::string(keyspace), keys_names,
                                 columns_names, std::string(token_range_pred), token_ranges, session, config);
    } catch (ModuleException e) {
        std::cerr << " ERROR CREATE TABLE" << table << " " << keyspace << " " << e.what() << std::endl;
    }
}




int Cache::disconnectCassandra() {
    if (session != NULL) {
        CassFuture *close_future = cass_session_close(session);
        cass_future_free(close_future);
        cass_session_free(session);
        cass_cluster_free(cluster);
        session = NULL;
    }
    return 0;
}



/*** HCACHE DATA TYPE METHODS AND SETUP ***/
/*
int Cache::put_row(void* keys, void* values) {
    this->T->put_crow(keys, values);
    return 0;
}


std::shared_ptr<void> Cache::get_row(void* keys){
    return this->T->get_crow(keys);
}
*/

/*** ITERATOR METHODS AND SETUP ***/

/*
Prefetch* Cache::get_keys_iterator(uint16_t prefetch_size) {
    return T->get_keys_iter(prefetch_size);
}


Prefetch* Cache::get_values_iterator(uint16_t prefetch_size) {
    return T->get_items_iter(prefetch_size);
}


Prefetch* Cache::get_items_iterator(uint16_t prefetch_size) {
    return T->get_values_iter(prefetch_size);
}
 */