#include "CacheTable.h"

#define default_cache_size 10


/***
 * Constructs a cache which takes and returns data encapsulated as pointers to TupleRow or PyObject
 * Follows Least Recently Used replacement strategy
 * @param size Max elements the cache will hold, afterwards replacement takes place
 * @param table Name of the table being represented
 * @param keyspace Name of the keyspace whose table belongs to
 * @param query Query ready to be bind with the keys
 * @param session
 */
CacheTable::CacheTable(const TableMetadata* table_meta, CassSession *session,
                       std::map<std::string,std::string> &config) {

    if (!session)
        throw ModuleException("CacheTable: Session is Null");

    int32_t cache_size = default_cache_size;

    if (config.find("cache_size")!=config.end()) {
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
    CHECK_CASS("CacheTable: Prepare query failed");
    this->prepared_query = cass_future_get_prepared(future);
    cass_future_free(future);
    this->myCache = NULL;
    this->session = session;
    this->table_metadata = table_meta;
    this->writer = new Writer(table_meta,session,config);
    this->keys_factory = new TupleRowFactory(table_meta->get_keys());
    this->values_factory = new TupleRowFactory(table_meta->get_values());
    if (cache_size) this->myCache = new Poco::LRUCache<TupleRow, TupleRow>(cache_size);
};


CacheTable::~CacheTable() {
    delete(writer);
    if (myCache) {
        //stl tree calls deallocate for cache nodes on clear()->erase(), and later on destroy, which ends up calling the deleters
        myCache->clear();
        delete (myCache);
    }
    delete (keys_factory);
    delete (values_factory);
    if (prepared_query!=NULL) cass_prepared_free(prepared_query);
    prepared_query=NULL;
    delete(table_metadata);
}


void CacheTable::put_crow(const TupleRow* keys, const TupleRow* values) {
    this->writer->write_to_cassandra(keys,values);
    if (myCache) this->myCache->update(*keys,values); //Inserts if not present, otherwise replaces
}


void CacheTable::put_crow(void* keys, void* values) {
    const TupleRow *k = keys_factory->make_tuple(keys);
    const TupleRow *v = values_factory->make_tuple(values);
    this->put_crow(k,v);
}


void CacheTable::put_crow(const TupleRow* row) {
    //split into two and call put_crow(a,b);
    std::shared_ptr<const std::vector<ColumnMeta> > keys_meta = keys_factory->get_metadata();
    uint16_t nkeys = (uint16_t)keys_meta->size();
    std::shared_ptr<const std::vector<ColumnMeta> > values_meta = values_factory->get_metadata();
    uint16_t nvalues = (uint16_t)values_meta->size();

    char* keys = (char*) malloc(keys_meta->at(nkeys-(uint16_t)1).position+keys_meta->at(nkeys-(uint16_t)1).size);
    char* values = (char*) malloc(values_meta->at(nvalues-(uint16_t)1).position+values_meta->at(nvalues-(uint16_t)1).size);


    for (uint16_t i=0; i<nkeys; ++i) {
        memcpy(keys+keys_meta->at(i).position,row->get_element(i),keys_meta->at(i).size);
    }
    for (uint16_t i=0; i<nvalues; ++i) {
        memcpy(values+values_meta->at(i).position,row->get_element(i+nkeys),values_meta->at(i).size);
    }
    this->put_crow(keys,values);
}

/*
 * POST: never returns NULL
 */
std::vector<const TupleRow *> CacheTable::retrieve_from_cassandra(const TupleRow *keys){

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
        throw ModuleException("CacheTable: Get row error on result"+error);
    }

    cass_future_free(query_future);
    cass_statement_free(statement);

    if (!cass_result_row_count(result))
        throw ModuleException("No rows found for this key");
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
    return values;
}


std::vector<const TupleRow *>  CacheTable::get_crow(const TupleRow *keys) {
    if (myCache) {
        Poco::SharedPtr<TupleRow> ptrElem = myCache->get(*keys);
        if (!ptrElem.isNull())  return std::vector <const TupleRow*> { ptrElem.get() };
    }

    std::vector<const TupleRow *> values = retrieve_from_cassandra(keys);

    if (myCache) myCache->add(*keys, values[0]);

    return values;
}

std::shared_ptr<void> CacheTable::get_crow(void* keys) {
    std::vector<const TupleRow*> result = get_crow(keys_factory->make_tuple(keys));
    if (result.empty()) return NULL;
    if (myCache) myCache->add(*keys_factory->make_tuple(keys), result[0]);
    return result.at(0)->get_payload();
}