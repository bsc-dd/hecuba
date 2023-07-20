#ifndef HFETCH_StorageInterface_H
#define HFETCH_StorageInterface_H

#include <vector>
#include <tuple>
#include <iostream>
#include <vector>
#include <unordered_map>


#include <cassandra.h>

#include "TableMetadata.h"
#include "Prefetch.h"
#include "Writer.h"
#include "CacheTable.h"

//#include "ArrayDataStore.h"

#include "configmap.h"


struct tokenHost {
    int64_t token;
    char *  host;
};

class StorageInterface {

public:

    StorageInterface(int nodePort, std::string contact_points, std::map<std::string, std::string>& config);

    ~StorageInterface();


    CacheTable *make_cache(const TableMetadata *table_meta,
                           config_map &config);


    CacheTable *make_cache(const char *table, const char *keyspace,
                           std::vector<config_map> &keys_names,
                           std::vector<config_map> &columns_names,
                           config_map &config);

    CacheTable *get_static_metadata_cache(config_map &config);


    Writer *make_writer(const char *table, const char *keyspace,
                        std::vector<config_map> &keys_names,
                        std::vector<config_map> &columns_names,
                        config_map &config);

    Writer *make_writer_stream(const char *table, const char *keyspace,
                                      std::vector<config_map> &keys_names,
                                      std::vector<config_map> &columns_names,
                                      const char* topic,
                                      config_map &config) ;

    void enable_writer_stream(Writer *target, const char *topic, config_map &config);

    Writer *make_writer(const TableMetadata *table_meta,
                        config_map &config);


    //ArrayDataStore *make_array_store(const char *table, const char *keyspace, config_map &config);

    Prefetch *get_iterator(const char *table, const char *keyspace,
                           std::vector<config_map> &keys_names,
                           std::vector<config_map> &columns_names,
                           const std::vector<std::pair<int64_t, int64_t>> &tokens,
                           config_map &config);

    Prefetch *get_iterator(const TableMetadata *table_meta,
                           const std::vector<std::pair<int64_t, int64_t>> &tokens,
                           config_map &config);

    Prefetch *get_iterator(const TableMetadata *table_meta,
                           config_map &config);

    inline CassSession *get_session() {
        if (!this->session) throw ModuleException("Cassandra not connected yet, session unavailable");
        return this->session;
    }

    int disconnectCassandra();

    char * get_host_per_token(int64_t token);
    std::vector<std::pair<int64_t,int64_t>> get_token_ranges() const;

private:

    std::vector< struct tokenHost > tokensInCluster;
    std::vector <std::pair<int64_t,int64_t>> token_ranges = {};

    CassSession *session;

    CassCluster *cluster;

    void query_tokens( const char * peer, const char* tokens, const char* table, const char * node, int nodePort);
    void set_tokens_per_host( const char * node, int nodePort);
    void get_tokens_per_host(std::vector< struct tokenHost > &tokensInCluster);
    void generate_token_ranges() ;

};


#endif //HFETCH_StorageInterface_H
