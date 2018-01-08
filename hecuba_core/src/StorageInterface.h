#ifndef HFETCH_StorageInterface_H
#define HFETCH_StorageInterface_H

#include <vector>
#include <tuple>
#include <iostream>
#include <vector>
#include <unordered_map>


#include "CacheTable.h"
#include <cassandra.h>
#include "Prefetch.h"
#include "Writer.h"
#include "TableMetadata.h"
#include "ClusterConfig.h"

class StorageInterface {

public:

    StorageInterface(int nodePort, std::string contact_points);

    ~StorageInterface();

    CacheTable *make_cache(const char *table, const char *keyspace,
                           std::vector<std::map<std::string, std::string>> &keys_names,
                           std::vector<std::map<std::string, std::string>> &columns_names,
                           std::map<std::string, std::string> &config);

    Writer *make_writer(const char *table, const char *keyspace,
                        std::vector<std::map<std::string, std::string>> &keys_names,
                        std::vector<std::map<std::string, std::string>> &columns_names,
                        std::map<std::string, std::string> &config);


    Writer *make_writer(const TableMetadata *table_meta,
                        std::map<std::string, std::string> &config);

    Prefetch *get_iterator(const char *table, const char *keyspace,
                           std::vector<std::map<std::string, std::string>> &keys_names,
                           std::vector<std::map<std::string, std::string>> &columns_names,
                           const std::vector<std::pair<int64_t, int64_t>> &tokens,
                           std::map<std::string, std::string> &config);

    Prefetch *get_iterator(const TableMetadata *table_meta,
                           const std::vector<std::pair<int64_t, int64_t>> &tokens,
                           std::map<std::string, std::string> &config);

    inline const CassSession *get_session() {
        return this->cluster->get_session();
    }

private:

    ClusterConfig *cluster;
};


#endif //HFETCH_StorageInterface_H
