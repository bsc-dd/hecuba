#ifndef HFETCH_StorageInterface_H
#define HFETCH_StorageInterface_H

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


    Prefetch *get_iterator(const char *table, const char *keyspace,
                           std::vector<std::map<std::string, std::string>> &keys_names,
                           std::vector<std::map<std::string, std::string>> &columns_names,
                           const std::vector<std::pair<int64_t, int64_t>> &tokens,
                           std::map<std::string, std::string> &config);

    Prefetch *get_iterator(const TableMetadata *table_meta,
                           const std::vector<std::pair<int64_t, int64_t>> &tokens,
                           std::map<std::string, std::string> &config);

    int disconnectCassandra();

private:


    CassSession *session;

    CassCluster *cluster;
};


#endif //HFETCH_StorageInterface_H
