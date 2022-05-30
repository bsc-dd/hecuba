#include "CacheTable.h"
#include "debug.h"

#define default_cache_size 0


/***
 * Constructs a cache which takes and returns data encapsulated as pointers to TupleRow or PyObject
 * Follows Least Recently Used replacement strategy
 * @param size Max elements the cache will hold, afterwards replacement takes place
 * @param table Name of the table being represented
 * @param keyspace Name of the keyspace whose table belongs to
 * @param query Query ready to be bind with the keys
 * @param session
 */
CacheTable::CacheTable(const TableMetadata *table_meta, CassSession *session,
                       std::map<std::string, std::string> &config) {

    if (!session)
        throw ModuleException("CacheTable: Session is Null");

    int32_t cache_size = default_cache_size;
    this->disable_timestamps = false;

    if (config.find("timestamped_writes") != config.end()) {
        std::string check_timestamps = config["timestamped_writes"];
        std::transform(check_timestamps.begin(), check_timestamps.end(), check_timestamps.begin(), ::tolower);
        if (check_timestamps == "false" || check_timestamps == "no")
            this->disable_timestamps = true;
    }

    if (config.find("cache_size") != config.end()) {
        std::string cache_size_str = config["cache_size"];
        try {
            cache_size = std::stoi(cache_size_str);
            if (cache_size < 0) throw ModuleException("Cache size value must be >= 0");
        }
        catch (std::exception &e) {
            std::string msg(e.what());
            msg += " Malformed value in config for cache_size";
            throw ModuleException(msg);
        }
    }


    /** Parse names **/
    CassFuture *future = cass_session_prepare(session, table_meta->get_select_query());
    CassError rc = cass_future_error_code(future);
    CHECK_CASS("CacheTable: Select row query preparation failed");
    this->prepared_query = cass_future_get_prepared(future);
    cass_future_free(future);
    future = cass_session_prepare(session, table_meta->get_delete_query());
    rc = cass_future_error_code(future);
    this->delete_query = cass_future_get_prepared(future);
    CHECK_CASS("CacheTable: Delete row query preparation failed");
    cass_future_free(future);
    this->myCache = NULL;
    this->session = session;
    this->table_metadata = table_meta;
    this->writer = new Writer(table_meta, session, config);
    this->keys_factory = new TupleRowFactory(table_meta->get_keys());
    this->values_factory = new TupleRowFactory(table_meta->get_values());
    this->row_factory = new TupleRowFactory(table_meta->get_items());
    this->timestamp_gen = new TimestampGenerator();
    this->writer->set_timestamp_gen(this->timestamp_gen);
    this->topic_name = nullptr;
    this->consumer = nullptr;
    this->kafka_conf = nullptr;

    if (cache_size) this->myCache = new KVCache<TupleRow, TupleRow>(cache_size);
};


CacheTable::~CacheTable() {
    delete (writer);
    if (myCache) {
        //stl tree calls deallocate for cache nodes on clear()->erase(), and later on destroy, which ends up calling the deleters
        myCache->clear();
        delete (myCache);
    }
    delete (keys_factory);
    delete (values_factory);
    if (prepared_query != NULL) cass_prepared_free(prepared_query);
    prepared_query = NULL;
    if (delete_query != NULL) cass_prepared_free(delete_query);
    delete_query = NULL;
    delete (table_metadata);
    if (topic_name) {
        free(topic_name);
        topic_name = nullptr;
    }
    if (consumer) {
        rd_kafka_destroy(consumer);
        consumer = nullptr;
    }
}


const void CacheTable::flush_elements() const {
    this->writer->flush_elements();
}

const void CacheTable::wait_elements() const {
    this->writer->wait_writes_completion();
}

void CacheTable::send_event(const TupleRow *keys, const TupleRow *values) {
    this->writer->send_event(keys, values);
    if (myCache) this->myCache->add(*keys, values); //Inserts if not present, otherwise replaces
}

void CacheTable::put_crow(const TupleRow *keys, const TupleRow *values) {
    this->writer->write_to_cassandra(keys, values);
    if (myCache) this->myCache->add(*keys, values); //Inserts if not present, otherwise replaces
}


void CacheTable::put_crow(void *keys, void *values) {
    const TupleRow *k = keys_factory->make_tuple(keys);
    const TupleRow *v = values_factory->make_tuple(values);
    this->put_crow(k, v);
    delete (k);
    delete (v);
}


/** this method only adds the data to the cache
 *  without making it persistent
 * @param keys
 * @param values
 */

void CacheTable::add_to_cache(void *keys, void *values) {
    const TupleRow *k = keys_factory->make_tuple(keys);
    const TupleRow *v = values_factory->make_tuple(values);
    if (myCache) this->myCache->add(*k, v);
    delete (k);
    delete (v);
}
void CacheTable::add_to_cache(const TupleRow  *keys, const TupleRow *values) {
    if (myCache) this->myCache->add(*keys, values);
}

Writer * CacheTable::get_writer() {
    return this->writer;
}
void  CacheTable::enable_stream(const char * topic_name, std::map<std::string, std::string> &config) {
    int size = strlen(topic_name)+1;
    this->topic_name = (char*) malloc(size);
    memcpy(this->topic_name, topic_name, size);
    this->stream_config = config; //Copy values

    /* Kafka conf... */
    char errstr[512];
    char hostname[128];
    rd_kafka_conf_t *conf = rd_kafka_conf_new();

    if (gethostname(hostname, sizeof(hostname))) {
        fprintf(stderr, "%% Failed to lookup hostname\n");
        exit(1);
    }

    // PRODUCER: Why do we need to set client.id????
    if (rd_kafka_conf_set(conf, "client.id", hostname,
                              errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
        fprintf(stderr, "%% %s\n", errstr);
        exit(1);
    }

    // CONSUMER: Why do we need to set group.id????
    if (rd_kafka_conf_set(conf, "group.id", "hecuba",
                errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
        fprintf(stderr, "%% %s\n", errstr);
        exit(1);
    }

	//yolandab:from the beginning: only for consumers
	if (rd_kafka_conf_set(conf, "auto.offset.reset", "beginning",
                errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
        fprintf(stderr, "%% %s\n", errstr);
        exit(1);
    }

    // Setting bootstrap.servers...
    if (config.find("kafka_names") == config.end()) {
        throw ModuleException("KAFKA_NAMES are not set. Use: 'host1:9092,host2:9092'");
    }
    std::string kafka_names = config["kafka_names"];

    if (rd_kafka_conf_set(conf, "bootstrap.servers", kafka_names.c_str(),
                              errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
        fprintf(stderr, "%% %s\n", errstr);
        exit(1);
    }
    this->kafka_conf = conf;
}

void  CacheTable::enable_stream_producer(void) {
    this->get_writer()->enable_stream(this->topic_name, this->stream_config);
}

void  CacheTable::enable_stream_consumer(void) {
    char errstr[512];

    /* Create Kafka consumer handle */
    rd_kafka_t *rk;
    if (!(rk = rd_kafka_new(RD_KAFKA_CONSUMER, this->kafka_conf,
                    errstr, sizeof(errstr)))) {
        fprintf(stderr, "%% Failed to create new consumer: %s\n", errstr);
        exit(1);
    }

    rd_kafka_resp_err_t err;
    rd_kafka_topic_partition_list_t* topics = rd_kafka_topic_partition_list_new(1);
    rd_kafka_topic_partition_list_add(topics, this->topic_name, RD_KAFKA_PARTITION_UA);

    if ((err = rd_kafka_subscribe(rk, topics))) {
        fprintf(stderr, "%% Failed to start consuming topics: %s\n", rd_kafka_err2str(err));
        exit(1);
    }

    rd_kafka_topic_partition_list_t* current_topics;
    bool finish = false;
    while(!finish) {
        err = rd_kafka_subscription(rk, &current_topics);
        if (err) {
            fprintf(stderr, "%% Failed to get topics: %s\n", rd_kafka_err2str(err));
            exit(1);
        }
        if (current_topics->cnt == 0) {
            fprintf(stderr, "%% Failed to get topics: NO ELEMENTS\n");
        } else{
            //fprintf(stderr, "%% I got you \n");
            //fprintf(stderr, "%% I got you %s\n", current_topics->elems[0].topic);
            finish = true;
        }

        rd_kafka_topic_partition_list_destroy(current_topics);
    }
    this->consumer = rk;

}

rd_kafka_message_t * CacheTable::kafka_poll(void) {
    bool finish = false;
    rd_kafka_message_t *rkmessage = NULL;
    while(! finish) {
        rkmessage = rd_kafka_consumer_poll(this->consumer, 500);
        if (rkmessage) {
            if (rkmessage->err) {
                    fprintf(stderr, "poll: error %s\n", rd_kafka_err2str(rkmessage->err));
            }else {
                finish=true;
            }
        } else {
            fprintf(stderr, "poll: timeout\n");
        }
    }
    return rkmessage;
}

// If we are receiving a numpy, we already have the memory allocated. We just need to copy on that memory the message received
void CacheTable::poll(char *data, const uint64_t size) {
    rd_kafka_message_t *rkmessage = this->kafka_poll();
    if (size != rkmessage->len) {
        char b[256];
        sprintf(b, "Expected numpy of size %ld, received a buffer of size %ld",size,rkmessage->len);
        throw ModuleException(b);
    }

    memcpy(data, rkmessage->payload, rkmessage->len);
    rd_kafka_message_destroy(rkmessage);
}

// If we are receiving a dictionary, we need to build the data structure to return and we need to add the key and the value to the cache
std::vector<const TupleRow *>  CacheTable::poll(void) {
    std::vector<const TupleRow *> result(1);

    rd_kafka_message_t *rkmessage = this->kafka_poll();

    char *keys = (char *) keys_factory->decode((void*)rkmessage->payload, rkmessage->len);
    TupleRow *k = keys_factory->make_tuple(keys); // Needed to calculate 'keyslength'
    uint64_t keyslength = keys_factory->get_content_size(k);

    char *keystoreturn = (char *) keys_factory->decode((void*)rkmessage->payload, rkmessage->len);

    uint64_t valueslength = rkmessage->len - keyslength;
    char *values = (char *) values_factory->decode((void*)((char*)rkmessage->payload+keyslength), valueslength);

    char *valuestoreturn = (char *)values_factory->decode((void*)((char*)rkmessage->payload+keyslength), valueslength);

            char myfinal[CASS_UUID_STRING_LENGTH];
            cass_uuid_string((**(CassUuid**)values), myfinal);
            DBG( "CacheTable::poll BEFORE" + std::string(myfinal));
    TupleRow *v = values_factory->make_tuple((void*)values);
            cass_uuid_string((**(CassUuid**)v->get_payload()), myfinal);
            DBG( "CacheTable::poll AFTER" + std::string(myfinal));

    add_to_cache(k, v); // Add key,value to cache

    char *row_buffer = (char*) malloc(k->length() + v->length());
    memcpy(row_buffer, keystoreturn, k->length());
    memcpy(row_buffer + k->length(), valuestoreturn, v->length());

    TupleRow *r = row_factory->make_tuple(row_buffer);
    result[0] = r; // Transform rkmessage to vector<tuplerow>
    rd_kafka_message_destroy(rkmessage);
    return result;
}

/*
 * POST: never returns NULL
 */
std::vector<const TupleRow *> CacheTable::retrieve_from_cassandra(const TupleRow *keys) {

    // To avoid consistency problems we flush the elements pending to be written
    this->writer->flush_elements();

    /* Not present on cache, a query is performed */
    CassStatement *statement = cass_prepared_bind(prepared_query);

    this->keys_factory->bind(statement, keys, 0);

    CassFuture *query_future = cass_session_execute(session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    CassError rc = cass_future_error_code(query_future);
    if (result == NULL) {
        /* Handle error */
        std::string error(cass_error_desc(rc));
        cass_future_free(query_future);
        cass_statement_free(statement);
        throw ModuleException("CacheTable: Get row error on result" + error);
    }

    cass_future_free(query_future);
    cass_statement_free(statement);

    uint32_t counter = 0;
    std::vector<const TupleRow *> values(cass_result_row_count(result));

    const CassRow *row;
    CassIterator *it = cass_iterator_from_result(result);
    while (cass_iterator_next(it)) {
        row = cass_iterator_get_row(it);
        values[counter] = values_factory->make_tuple(row);
        ++counter;
    }
    cass_iterator_free(it);
    cass_result_free(result);
    return values;
}


std::vector<const TupleRow *> CacheTable::get_crow(const TupleRow *keys) {

    if (myCache) {
        TupleRow *value;
        try {
            value = new TupleRow(myCache->get(*keys));
            return std::vector<const TupleRow *>{value};
        }
        catch (std::out_of_range &ex) {
            value = nullptr;
        }
    }

    std::vector<const TupleRow *> values = retrieve_from_cassandra(keys);

    if (myCache && !values.empty()) myCache->add(*keys, values[0]);

    return values;
}


std::vector<const TupleRow *> CacheTable::get_crow(void *keys) {
    const TupleRow *tuple_key = keys_factory->make_tuple(keys);
    std::vector<const TupleRow *> result = get_crow(tuple_key);
    delete (tuple_key);
    return result;
}


void CacheTable::delete_crow(const TupleRow *keys) {

    //Remove row from Cassandra
    CassStatement *statement = cass_prepared_bind(delete_query);

    this->keys_factory->bind(statement, keys, 0);
    if (disable_timestamps) this->writer->flush_elements();
    else cass_statement_set_timestamp(statement, timestamp_gen->next()); // Set delete time

    CassFuture *query_future = cass_session_execute(session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    CassError rc = cass_future_error_code(query_future);
    if (result == NULL) {
        /* Handle error */
        std::string error(cass_error_desc(rc));
        cass_future_free(query_future);
        cass_statement_free(statement);
        throw ModuleException("CacheTable: Delete row error on result" + error);
    }

    cass_future_free(query_future);
    cass_statement_free(statement);
    cass_result_free(result);

    //Remove entry from cache
    if (myCache) myCache->remove(*keys);
}
