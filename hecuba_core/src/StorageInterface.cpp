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

    // Query tokens
    std::string node = contact_points.substr(0, contact_points.find_first_of(","));
    set_tokens_per_host(node.c_str(), nodePort);

    printf("-----------------\n");
    for (int i = 0; i < tokensInCluster.size(); i++) {
                printf("token: %ld %s\n", tokensInCluster[i].token, tokensInCluster[i].host);
    }
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


CacheTable *StorageInterface::make_cache(const TableMetadata *table_meta,
                                         config_map &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    return new CacheTable(table_meta, session, config);
}


CacheTable *StorageInterface::make_cache(const char *table, const char *keyspace,
                                         std::vector<config_map> &keys_names,
                                         std::vector<config_map> &columns_names,
                                         config_map &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
    return new CacheTable(table_meta, session, config);
}


Writer *StorageInterface::make_writer(const char *table, const char *keyspace,
                                      std::vector<config_map> &keys_names,
                                      std::vector<config_map> &columns_names,
                                      config_map &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
    return new Writer(table_meta, session, config);

}


Writer *StorageInterface::make_writer(const TableMetadata *table_meta,
                                      config_map &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    return new Writer(table_meta, session, config);

}


//ArrayDataStore *StorageInterface::make_array_store(const char *table, const char *keyspace, config_map &config) {
//    return new ArrayDataStore(table, keyspace, session, config);
//}


MetaManager *StorageInterface::make_meta_manager(const char *table, const char *keyspace,
                                                 std::vector<config_map> &keys_names,
                                                 std::vector<config_map> &columns_names,
                                                 config_map &config) {
    if (!session){
        std::cerr<<"StorageInterface not connected to any node"<<std::endl;
        throw ModuleException("StorageInterface not connected to any node");
    }
 
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
    return new MetaManager(table_meta, session, config);
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
                                         std::vector<config_map> &keys_names,
                                         std::vector<config_map> &columns_names,
                                         const std::vector<std::pair<int64_t, int64_t>> &tokens,
                                         config_map &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
    return new Prefetch(tokens, table_meta, session, config);
}


Prefetch *StorageInterface::get_iterator(const TableMetadata *table_meta,
                                         const std::vector<std::pair<int64_t, int64_t>> &tokens,
                                         config_map &config) {
    if (!session) throw ModuleException("StorageInterface not connected to any node");
    return new Prefetch(tokens, table_meta, session, config);
}
/* Query 'peer' and 'tokens' columns from 'table' at 'node:nodePort'
 * Results are stored in 'tokensInCluster' field */
void StorageInterface::query_tokens( const char * peer, const char* tokens, const char* table, const char * node, int nodePort) {
    char stmnt[80];

    sprintf(stmnt, "Select %s, %s from %s", peer, tokens, table);
	CassStatement* statement  = cass_statement_new(stmnt,0);
    // Pick the firs contact_point
    if (cass_statement_set_host(statement, node, nodePort) != CASS_OK) {
        write(2, "ooops", 5);
        write(2, "\n",1);
        exit(1);
    }
	CassFuture* query_future = cass_session_execute(session, statement);

    cass_statement_free(statement);

    const CassResult* result = cass_future_get_result(query_future);
    if (result == NULL) {
        CassError error_code = cass_future_error_code(query_future);
        const char* error_message;
        size_t error_message_length;
        cass_future_error_message(query_future, &error_message, &error_message_length);
        write(2, error_message, error_message_length);
        write(2, "\n",1);
        exit(1);
    }

    cass_future_free(query_future);

    CassIterator* row_iterator = cass_iterator_from_result(result);

    while (cass_iterator_next(row_iterator)) {
        const CassRow* row = cass_iterator_get_row(row_iterator);
        CassInet peer_n;
        char * str_peer = (char*) malloc(sizeof(char*)*80);
        bool found = false;
        cass_value_get_inet(cass_row_get_column_by_name(row, peer), &peer_n);
        cass_inet_string(peer_n, str_peer);
        //printf("peer: %s\n",str_peer);

        const CassValue* col = cass_row_get_column_by_name(row, tokens);
        if (cass_value_is_collection(col)) {
            uint32_t rows = 0;
            CassIterator* col_it = cass_iterator_from_collection(col);
            while(cass_iterator_next(col_it)) {
                const CassValue *item = cass_iterator_get_value(col_it);
                const char *str;
                size_t size;
                cass_value_get_string(item, &str, &size);
                //printf("token: %s %s\n", str, str_peer);
                rows++;

                struct tokenHost x;
                x.token = strtoll(str, NULL, 10);
                x.host  = str_peer;
                std::vector< struct tokenHost >::iterator i;
                found = false;
                i = tokensInCluster.begin();
                while (!found && i < tokensInCluster.end()){
                    if (x.token < (*i).token) {
                        tokensInCluster.insert(i, x);
                        found = true;
                    }
                    i++;
                }
                if (!found) {
                    tokensInCluster.push_back(x);
                }
            }
            cass_iterator_free(col_it);
            //printf("num tokens: %u\n", rows);
        } else {
            write(2, "tokens is not a collection\n",26);
            exit(1);
        }
    }

    cass_result_free(result);
    cass_iterator_free(row_iterator);

}

/* tokensInCluster setter
 * To enforce the execution of all queries to the same node, node and nodePort are required */
void StorageInterface::set_tokens_per_host(const char * node, int nodePort) {
    query_tokens("listen_address", "tokens", "system.local", node, nodePort);
    query_tokens("peer"          , "tokens", "system.peers", node, nodePort);
}

void StorageInterface::get_tokens_per_host(std::vector< struct tokenHost > &tokensInCluster) {
    tokensInCluster = this->tokensInCluster;
}

/* Returns the host associated to a 'token' */
char * StorageInterface::get_host_per_token(int64_t token) {
    std::cout<<"JCOSTA =================" << std::endl;
    struct tokenHost *th = NULL; //tokensInCluster.find(token)
    bool found = false;
    uint32_t i=0;
    for (; (i < tokensInCluster.size() - 1) && (!found); i++) {
        //printf("token: %ld %s\n", tokensInCluster[i].token, tokensInCluster[i].host);
        if (tokensInCluster[i].token   <= token &&
            tokensInCluster[i+1].token >  token) {
            found = true;
            th = &tokensInCluster[i];
        }
    }
    if (!found) {
        th = &tokensInCluster[i];
    }
    printf("JCOSTA token found %ld -> %ld %s\n", token, th->token, th->host);
    return th->host;
}
