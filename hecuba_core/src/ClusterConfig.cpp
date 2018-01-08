#include "ClusterConfig.h"


ClusterConfig::ClusterConfig() : ClusterConfig(node_port, contact_names){

}

ClusterConfig::ClusterConfig(int32_t port, std::string &points) {
    this->node_port = port;
    this->contact_names = points;

    CassFuture *connect_future = nullptr;
    cluster = cass_cluster_new();
    session = cass_session_new();

    // add contact points
    if (contact_names.empty()) contact_names = "127.0.0.1";
    cass_cluster_set_contact_points(cluster, contact_names.c_str());
    cass_cluster_set_port(cluster, node_port);
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

ClusterConfig::~ClusterConfig() {
    if (session != nullptr) {
        CassFuture *close_future = cass_session_close(session);
        cass_future_free(close_future);
        cass_session_free(session);
        cass_cluster_free(cluster);
        session = nullptr;
    }
}


void ClusterConfig::set_contact_port(int32_t port) {
    this->node_port = port;
}

void ClusterConfig::set_contact_names(std::string &points) {
    this->contact_names = points;
}

std::string ClusterConfig::get_contact_names() const {
    return this->contact_names;
}

int32_t ClusterConfig::get_contact_port() const {
    return this->node_port;
}