#ifndef CPP_INTERFACE_CLUSTERCONFIG_H
#define CPP_INTERFACE_CLUSTERCONFIG_H

#include <iostream>


#include <cassandra.h>
#include "ModuleException.h"


class ClusterConfig {
public:

    ClusterConfig();

    ClusterConfig(int32_t port, std::string &points);

    ~ClusterConfig();

    inline CassSession* get_session() const {
        return session;
    };

    void set_contact_port(int32_t port);

    void set_contact_names(std::string &points);


    int32_t get_contact_port() const;

    std::string get_contact_names() const;

private:
    std::string contact_names;
    int32_t node_port;
    CassSession *session;
    CassCluster *cluster;
};


#endif //CPP_INTERFACE_CLUSTERCONFIG_H
