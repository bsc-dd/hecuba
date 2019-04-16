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

#include "ArrayDataStore.h"

typedef std::map <std::string, std::string> config_map;


class StorageInterface {

public:

    StorageInterface(int nodePort, std::string contact_points);

    ~StorageInterface();


    CacheTable *make_cache(const TableMetadata *table_meta,
                           config_map &config);


    CacheTable *make_cache(const char *table, const char *keyspace,
                           std::vector<config_map> &keys_names,
                           std::vector<config_map> &columns_names,
                           config_map &config);

    Writer *make_writer(const char *table, const char *keyspace,
                        std::vector<config_map> &keys_names,
                        std::vector<config_map> &columns_names,
                        config_map &config);


    Writer *make_writer(const TableMetadata *table_meta,
                        config_map &config);


    ArrayDataStore *make_arrray_store(const char *table, const char *keyspace,
                        std::vector<config_map> &keys_names,
                        std::vector<config_map> &columns_names,
                        config_map &config);

    Prefetch *get_iterator(const char *table, const char *keyspace,
                           std::vector<config_map> &keys_names,
                           std::vector<config_map> &columns_names,
                           const std::vector<std::pair<int64_t, int64_t>> &tokens,
                           config_map &config);

    Prefetch *get_iterator(const TableMetadata *table_meta,
                           const std::vector<std::pair<int64_t, int64_t>> &tokens,
                           config_map &config);

    inline const CassSession *get_session() {
        return this->session;
    }

    int disconnectCassandra();

private:


    CassSession *session;

    CassCluster *cluster;
};


#endif //HFETCH_StorageInterface_H
