#ifndef CPP_INTERFACE_CLUSTERCONFIG_H
#define CPP_INTERFACE_CLUSTERCONFIG_H

#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include <vector>
#include <unordered_map>


#include "CacheTable.h"
#include "Poco/LRUCache.h"
#include <cassandra.h>
#include "Prefetch.h"
#include "Writer.h"
#include "TableMetadata.h"

class ClusterConfig {
public:

    ClusterConfig();

    ClusterConfig(int32_t port, std::string &points);

    ~ClusterConfig();

    void set_contact_port(int32_t port);
    void set_contact_names(std::string &points);

    CassSession* get_session() {
        return session;
    };

    std::string get_contact_names();

private:
    std::string contact_names;
    int32_t node_port;
    CassSession *session;
    CassCluster *cluster;
};


#endif //CPP_INTERFACE_CLUSTERCONFIG_H
