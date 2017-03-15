#ifndef HFETCH_Cache_H
#define HFETCH_Cache_H

#include <vector>
#include "CacheTable.h"
#include <tuple>
#include <iostream>
#include <vector>
#include <unordered_map>

#include "Poco/LRUCache.h"
#include <cassandra.h>
#include "Prefetch.h"



class Cache {

public:

    Cache(int nodePort, std::string contact_points);

    ~Cache();

    CacheTable* makeCache(const char *table,const char *keyspace, std::vector < std::string> keys_names,std::vector < std::string > columns_names, std::vector<std::pair<int64_t, int64_t>> token_ranges,std::map<std::string,std::string> &config);

    /*int put_row(void* keys, void* values);

    std::shared_ptr<void> get_row(void* keys);

    Prefetch* get_keys_iterator(uint16_t prefetch_size);

    Prefetch* get_values_iterator(uint16_t prefetch_size);

    Prefetch* get_items_iterator(uint16_t prefetch_size);
*/
private:

    int disconnectCassandra();

    CassSession* session;

    CassCluster *cluster;
};


#endif //HFETCH_Cache_H
