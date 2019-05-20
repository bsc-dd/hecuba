#include "CacheTable.h"

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
            disable_timestamps = true;
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
    this->timestamp_gen = TimestampGenerator();
    this->writer->set_timestamp_gen(this->timestamp_gen);
    if (cache_size) this->myCache = new TupleRowCache<TupleRow, TupleRow>(cache_size);
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
}


const void CacheTable::flush_elements() const {
    this->writer->flush_elements();
}

void CacheTable::put_crow(const TupleRow *keys, const TupleRow *values) {
    this->writer->write_to_cassandra(keys, values);
    if (myCache) this->myCache->update(*keys, values); //Inserts if not present, otherwise replaces
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
    if (myCache) this->myCache->update(*k, v);
    delete (k);
    delete (v);
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
        Poco::SharedPtr<TupleRow> ptrElem = myCache->get(*keys);
        if (!ptrElem.isNull()) return std::vector<const TupleRow *>{new TupleRow(ptrElem.get())};
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
    else cass_statement_set_timestamp(statement, timestamp_gen.next()); // Set delete time

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
    if (myCache) myCache->Remove(*keys);
}
