#include "CacheTable.h"
#include "debug.h"
#include "unistd.h"
#include "HecubaExtrae.h"

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
                       std::map<std::string, std::string> &config, bool free_table_meta) {

    //std::cout<< "CacheTable::CacheTable "<< table_meta->get_table_name()<<"."<<table_meta->get_keyspace()<<" free:"<<free_table_meta<<std::endl;
    DBG( "CacheTable::CacheTable "<< table_meta->get_table_name()<<"."<<table_meta->get_keyspace()<<" free:"<<free_table_meta << " @" << std::hex<< this) ;
    if (!session)
        throw ModuleException("CacheTable: Session is Null");

    int32_t cache_size = default_cache_size;
    this->disable_timestamps = false;

    if (config.find("timestamped_writes") != config.end()) {
        std::string check_timestamps = config["timestamped_writes"];
        for (long unsigned int i = 0; i < check_timestamps.size(); i++) {
            check_timestamps[i] = ::tolower(check_timestamps[i]);
        }
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
    HecubaExtrae_event(HECUBACASS, HBCASS_PREPARES);
    CassFuture *future = cass_session_prepare(session, table_meta->get_select_query());
    CassError rc = cass_future_error_code(future);
    CHECK_CASS("CacheTable: Select row query preparation failed " + table_meta->get_select_query());
    this->prepared_query = cass_future_get_prepared(future);
    cass_future_free(future);
    future = cass_session_prepare(session, table_meta->get_delete_query());
    rc = cass_future_error_code(future);
    this->delete_query = cass_future_get_prepared(future);
    CHECK_CASS("CacheTable: Delete row query preparation failed");
    cass_future_free(future);
    HecubaExtrae_event(HECUBACASS, HBCASS_END);
    this->myCache = NULL;
    this->session = session;
    this->table_metadata = table_meta;
    this->writer = new Writer(table_meta, session, config);
    this->keys_factory = new TupleRowFactory(table_meta->get_keys());
    this->values_factory = new TupleRowFactory(table_meta->get_values());
    this->row_factory = new TupleRowFactory(table_meta->get_items());
    HecubaExtrae_event(HECUBADBG, HECUBA_TIMESTAMPGENERATOR);
    this->timestamp_gen = new TimestampGenerator();
    this->writer->set_timestamp_gen(this->timestamp_gen);
    HecubaExtrae_event(HECUBADBG, HECUBA_END);

    this->kafka_conf = nullptr;
    this->should_table_meta_be_freed = free_table_meta;

    HecubaExtrae_event(HECUBADBG, HECUBA_KVCACHE);
    if (cache_size) this->myCache = new KVCache<TupleRow, TupleRow>(cache_size);
    HecubaExtrae_event(HECUBADBG, HECUBA_END);
};

CacheTable::CacheTable(const CacheTable& src) {
	DBG(" Copy operator ");
    *this = src;
}

CacheTable& CacheTable::operator = (const CacheTable& src) {
	DBG(" Assignment operator ");
    if (this != &src) {
        this->session = src.session;
        if (this->table_metadata!=nullptr) { delete (this->table_metadata); }
        this->table_metadata = new TableMetadata(*src.table_metadata);
        CassFuture *future = cass_session_prepare(session, table_metadata->get_select_query());
        CassError rc = cass_future_error_code(future);
        CHECK_CASS("CacheTable: Select row query preparation failed" + table_metadata->get_select_query());
        this->prepared_query = cass_future_get_prepared(future);
        cass_future_free(future);
        future = cass_session_prepare(session, table_metadata->get_delete_query());
        rc = cass_future_error_code(future);
        this->delete_query = cass_future_get_prepared(future);
        CHECK_CASS("CacheTable: Delete row query preparation failed");
        cass_future_free(future);
        this->myCache = NULL;
        if (this->writer!=nullptr) { delete (this->writer); }
        this->writer = new Writer(*src.writer);
        if (this->keys_factory != nullptr) {delete (this->keys_factory);}
        this->keys_factory = new TupleRowFactory(table_metadata->get_keys()); // TODO check if TupleRowFactory implements copy assignment: integer and vector of ColumnMeta.... I guess it is not necessary to instantiate a new one
        if (this->values_factory != nullptr) {delete (this->values_factory);}
        this->values_factory = new TupleRowFactory(table_metadata->get_values());
        if (this->row_factory != nullptr) {delete (this->row_factory);}
        this->row_factory = new TupleRowFactory(table_metadata->get_items());
        if (this->timestamp_gen != nullptr) {delete (this->timestamp_gen);}
        this->timestamp_gen = new TimestampGenerator();
        this->writer->set_timestamp_gen(this->timestamp_gen);
        if (this->kafka_conf != nullptr) { free(this->kafka_conf); }
        if (src.kafka_conf != nullptr) { this->kafka_conf = rd_kafka_conf_dup(src.kafka_conf); }
        for (auto const &x:kafkaConsumer) {
            free(x.second); //deallocate consumer
            kafkaConsumer.erase(x.first);
        }
        for (auto const &x:src.kafkaConsumer) {
            enable_stream_consumer(x.first.c_str()); // is it possible to delay this?
            enable_stream_producer(x.first.c_str());
        }
        if (this->myCache !=nullptr) {delete(this->myCache);}
        if (src.myCache != NULL) this->myCache = new KVCache<TupleRow, TupleRow>(src.myCache->get_max_cache_size());
        this->should_table_meta_be_freed = src.should_table_meta_be_freed;
    }

    return *this;
}

CacheTable::~CacheTable() {
    DBG( " Destructor  "<< table_metadata->get_table_name()<<"."<<table_metadata->get_keyspace()<<" free:"<< should_table_meta_be_freed<< " @" << std::hex<< this) ;
    if (this->writer->is_stream_out_enable()) { 
       for (auto const &x: this->writer->getKafkaTopics()){
           close_stream (x.first.c_str());
       } 
    }
    //yolandab: los topics son los de writer, si no se ha inicializado para escribir no hay que hacer close_stream
#if 0
    for (auto const &x: kafkaConsumer) {
	std::cout<<"~CacheTable: close_stream: " << x.first.c_str()<<std::endl;
       close_stream(x.first.c_str()); //send EOD to the consumer before deleting the writer
    }
#endif
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
    DBG( this<< " table_metadata = "<< table_metadata);
    if (table_metadata != nullptr) {
        if (should_table_meta_be_freed) {
            delete (table_metadata);
        }
        table_metadata = nullptr;
    }
    for (auto const &x:kafkaConsumer) {
        rd_kafka_destroy(x.second);
    }
    kafkaConsumer.clear();
}


const void CacheTable::flush_elements() const {
    DBG(" Flushing "<<get_metadata()->get_table_name());
    this->writer->flush_elements();
}

const void CacheTable::wait_elements() const {
    DBG(" Waiting "<<get_metadata()->get_table_name());
    this->writer->wait_writes_completion();
}

void CacheTable::send_event(const char* topic_name, const TupleRow *keys, const TupleRow *values) {
    this->writer->send_event(topic_name, keys, values);
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

/* Generate a new TupleRow for the 'keys' pointer.
 * The resulting TupleRow must be deleted when not needed!
 * This is needed when you want to reuse a TupleRow (or more apropiately, the
 * memory pointed by 'keys' as the usual method creates a new TupleRow and
 * DELETES the content after the use... which makes the pointer content
 * useless)*/
const TupleRow* CacheTable::get_new_keys_tuplerow(void* keys) const {
    return keys_factory->make_tuple(keys);
}

/* Generate a new TupleRow for the 'values' pointer.
 * The resulting TupleRow must be deleted when not needed!
 * This is needed when you want to reuse a TupleRow (or more apropiately, the
 * memory pointed by 'values' as the usual method creates a new TupleRow and
 * DELETES the content after the use... which makes the pointer content
 * useless)*/
const TupleRow* CacheTable::get_new_values_tuplerow(void* values) const {
    return values_factory->make_tuple(values);
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
    if (myCache) this->myCache->add(keys, values);
}

Writer * CacheTable::get_writer() {
    return this->writer;
}
void  CacheTable::enable_stream(std::map<std::string, std::string> &config) {
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

void  CacheTable::enable_stream_producer(const char* topic_name) {
    this->get_writer()->enable_stream(topic_name, this->stream_config);
}

void  CacheTable::enable_stream_consumer(const char* topic_name) {
    char errstr[512];
    std::string topic = std::string(topic_name);

    if (kafkaConsumer.find(topic) != kafkaConsumer.end()) { // Topic already Exists
        throw ModuleException(" Ooops. Stream "+topic+" already initialized.");
    } else {
        /* Create Kafka consumer handle (1 consumer per topic) */
        rd_kafka_t *rk;
        if (!(rk = rd_kafka_new(RD_KAFKA_CONSUMER, this->kafka_conf, errstr, sizeof(errstr)))) {
            fprintf(stderr, "%% Failed to create new consumer: %s\n", errstr);
            exit(1);
        }
	std::cout << "[ENRIC KAFKA] CacheTable::enable_stream_consumer; rd_kafka_new" << std::endl;

	rd_kafka_poll_set_consumer(rk); //enric
	std::cout << "[ENRIC KAFKA] CacheTable::enable_stream_consumer; rd_kafka_poll_set_consumer" << std::endl;


        rd_kafka_resp_err_t err;
        rd_kafka_topic_partition_list_t* topics = rd_kafka_topic_partition_list_new(1);
	std::cout << "[ENRIC KAFKA] CacheTable::enable_stream_consumer; rd_kafka_topic_partition_list_new" << std::endl;
        //rd_kafka_topic_partition_list_add(topics, topic_name, RD_KAFKA_PARTITION_UA);
        rd_kafka_topic_partition_list_add(topics, topic_name, 0);
	std::cout << "[ENRIC KAFKA] CacheTable::enable_stream_consumer; rd_kafka_topic_partition_list_add" << std::endl;

/*
        if ((err = rd_kafka_subscribe(rk, topics))) {
            fprintf(stderr, "%% Failed to start consuming topics: %s\n", rd_kafka_err2str(err));
            exit(1);
        }
	std::cout << "[ENRIC KAFKA] CacheTable::enable_stream_consumer; rd_kafka_subscribe" << std::endl;
*/

        rd_kafka_assign(rk, topics);
        /*
        rd_kafka_topic_partition_list_t* current_topics;
        bool finish = false;
        while(!finish) {
            err = rd_kafka_subscription(rk, &current_topics);
	std::cout << "[ENRIC KAFKA] CacheTable::enable_stream_consumer; rd_kafka_subscription" << std::endl;
            if (err) {
                fprintf(stderr, "%% Failed to get topics: %s\n", rd_kafka_err2str(err));
                exit(1);
            }
            if (current_topics->cnt == 0) {
                fprintf(stderr, "%% Failed to get topics: NO ELEMENTS\n");
            } else{
                //fprintf(stderr, "%% I got you \n");
                // fprintf(stderr, "%% I got you %s\n", current_topics->elems[0].topic);
                finish = true;
            }

            rd_kafka_topic_partition_list_destroy(current_topics);
        }
        kafkaConsumer[topic] = rk;
    }

}

rd_kafka_message_t * CacheTable::kafka_poll(const char* topic_name) {
    bool finish = false;
    rd_kafka_t *consumer;
    try {
        consumer=kafkaConsumer.at(std::string(topic_name));
    } catch (std::out_of_range &e) {
        throw ModuleException(" Ooops. Stream "+std::string(topic_name)+" not initialized.");
    }
    rd_kafka_message_t *rkmessage = NULL;
    while(! finish) {
        rkmessage = rd_kafka_consumer_poll(consumer, 500);
        if (rkmessage) {
            if (rkmessage->err) {
                    fprintf(stderr, "poll topic[%s]: error %s\n", topic_name, rd_kafka_err2str(rkmessage->err));
            }else {
                finish=true;
            }
        } else {
            fprintf(stderr, "poll topic[%s] : Nothing available after 500ms. Retrying.\n", topic_name);
        }
    }
    return rkmessage;
}

// If we are receiving a numpy, we already have the memory allocated. We just need to copy on that memory the message received
void CacheTable::poll(const char *topic_name, char *data, const uint64_t size) {
    uint64_t offset = 0;

    while (offset < size) {
        rd_kafka_message_t *rkmessage = this->kafka_poll(topic_name);

        if (size < (rkmessage->len+offset)) {
            char b[256];
            sprintf(b, "Expected numpy of size %ld, received buffers for a total size of %ld",size,rkmessage->len+offset);
            throw ModuleException(b);
        }

        memcpy(&data[offset], rkmessage->payload, rkmessage->len);
        rd_kafka_message_destroy(rkmessage);

        offset += rkmessage->len;
    }
}

// If we are receiving a dictionary, we need to build the data structure to return and we need to add the key and the value to the cache
std::vector<const TupleRow *>  CacheTable::poll(const char *topic_name) {
    std::vector<const TupleRow *> result(1);

    rd_kafka_message_t *rkmessage = this->kafka_poll(topic_name);
    DBG("CacheTable::poll: after kafka_poll" << topic_name);

    // The received message contains a sequence containing:
    //      key_nullvalues + key + value_nullvalues + value

    TupleRow *k = keys_factory->decode((void*)rkmessage->payload);

    uint64_t keyslength = keys_factory->get_content_size(k);
    uint32_t keynullvalues_size = std::ceil(((double)k->n_elem())/32)*sizeof(uint32_t);

    uint64_t valueslength = rkmessage->len - keyslength;
    TupleRow *v = values_factory->decode((void*)((char*)rkmessage->payload + keyslength + keynullvalues_size));

    // Create a Null values vector for the whole row
    std::vector<uint32_t> row_nullvalues(ceil((double)(k->n_elem()+v->n_elem())/32), 0);

    bool is_key_all_null=true;

    // Copy Null values for keys AND values to the row null-values vector
    uint32_t i;
    for (i = 0; i < k->n_elem(); i++){ // ... keys first
        if (k->isNull(i)) {
            row_nullvalues[i>>5] |= (1 << i%32);
        } else {
            is_key_all_null = false;
        }
    }
    for (uint32_t j = i; j< v->n_elem() + i; j++){ // ... then values
        if (v->isNull(j-i)) {
            row_nullvalues[j>>5] |= (1 << j%32);
        }
    }
    if (!is_key_all_null) {
        add_to_cache(k, v); // Add key,value to cache only if the key is not null
    } else {
        DBG("CacheTable::poll: all values in key are null");
    }

    // Finally, create the TupleRow to be returned
    char *row_buffer = (char*) malloc(k->length() + v->length());
    memcpy(row_buffer, (char*) k->get_payload(), k->length());
    memcpy(row_buffer + k->length(), (char*) v->get_payload(), v->length());

    TupleRow *r = row_factory->make_tuple(row_buffer);

    DBG("k nullvalues : "<< std::hex<< k->get_null_values()[0]);
    DBG("v nullvalues : "<< std::hex<< v->get_null_values()[0]);
    DBG("r nullvalues : "<< std::hex<< row_nullvalues[0]);
    DBG("r nullvalues.size() : "<< row_nullvalues.size() << " to store "<< (k->n_elem()+v->n_elem())<<"elements");

    //concatenate the null values info from the keys and the columns
    DBG("CacheTable::poll: before inserting null_values from columns" << topic_name);

    r->set_null_values(row_nullvalues);

    result[0] = r; // Transform rkmessage to vector<tuplerow>
    rd_kafka_message_destroy(rkmessage);
    DBG("CacheTable::poll. Before returning");
    return result;
}

/* disable_send_EOD: Avoid sending an EOD at close_stream. Useful for StorageNumpys */
void CacheTable::disable_send_EOD(void) {
	send_EOD=false;
}

/*
 * Sends an EOD to to the topic (basically a couple of TupleRows with all its
 * elements to NULL)
 */
void  CacheTable::close_stream(const char *topic_name) {
    // Create empty TupleRows for Keys and Values
    if (this->writer != nullptr){
    if (this->writer->is_stream_out_enable()){
	    if (send_EOD) {
    	uint64_t keyslength = keys_factory->get_nbytes();
    	char * keys_b = (char*)malloc(keyslength);
    	TupleRow *k = keys_factory->make_tuple(keys_b);
    	for (uint32_t i = 0; i < k->n_elem(); i++) {
        	k->setNull(i);
    	}

    	uint64_t valueslength = values_factory->get_nbytes();
    	char * values_b = (char*)malloc(valueslength);
    	TupleRow *v = values_factory->make_tuple(values_b);
    	for (uint32_t i = 0; i < v->n_elem(); i++) {
        	v->setNull(i);
    	}

    	// Send EOD
    	this->writer->send_event(topic_name, k, v);
	    }
    }
    }
}


/*
 * attr_name: Retrieve ONLY the column 'attr_name' from the row
 * POST: never returns NUL
 */
std::vector<const TupleRow *> CacheTable::retrieve_from_cassandra(const TupleRow *keys, const char* attr_name) {

    // To avoid consistency problems we flush the elements pending to be written
    flush_elements();

    /* Not present on cache, a query is performed */
    CassStatement *statement = cass_prepared_bind(prepared_query);

    this->keys_factory->bind(statement, keys, 0);

    HecubaExtrae_event(HECUBACASS, HBCASS_READ);
    CassFuture *query_future = cass_session_execute(session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    HecubaExtrae_event(HECUBACASS, HBCASS_END);
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
        if (attr_name) {
            const CassValue *val = cass_row_get_column_by_name(row, attr_name);
            TupleRowFactory * v_single_factory = new TupleRowFactory(table_metadata->get_single_value(attr_name));
            values[counter] = v_single_factory->make_tuple(val);
            delete (v_single_factory);
        } else {
            values[counter] = values_factory->make_tuple(row);
        }
        ++counter;
    }
    cass_iterator_free(it);
    cass_result_free(result);
    return values;
}

/*
 * attr_name: Retrieve ONLY the column 'attr_name' from the row
 * POST: never returns NUL
 */
std::vector<const TupleRow *> CacheTable::retrieve_from_cassandra(void *keys, const char* attr_name) {
    const TupleRow *tuple_key = keys_factory->make_tuple(keys);
    std::vector<const TupleRow *> result = retrieve_from_cassandra(tuple_key, attr_name);
    delete (tuple_key);
    return result;
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
    if (disable_timestamps) flush_elements();
    else cass_statement_set_timestamp(statement, timestamp_gen->next()); // Set delete time

    HecubaExtrae_event(HECUBACASS, HBCASS_DELETE);
    CassFuture *query_future = cass_session_execute(session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    HecubaExtrae_event(HECUBACASS, HBCASS_END);
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
