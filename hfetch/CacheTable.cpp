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

    //check session!=NULL
    if (!session)
        throw ModuleException("CacheTable: Session is Null");

    int32_t cache_size = default_cache_size;

    if (config.find("cache_size")!=config.end()) {
        std::string cache_size_str = config["cache_size"];
        try {
            cache_size = std::stoi(cache_size_str);
        }
        catch (std::exception &e) {
            std::string msg(e.what());
            msg+= " Malformed value in config for cache_size";
            throw ModuleException(msg);
        }
    }

    if (cache_size<0) throw ModuleException("Cache size value must be >= 0");


    CassFuture *future = cass_session_prepare(session, table_meta->get_select_query());
    CassError rc = cass_future_error_code(future);
    if (rc!=CASS_OK) {
        std::string error = cass_error_desc(rc);
        std::string query_text(table_meta->get_select_query());
        throw ModuleException("CacheTable: Preparing query: "+query_text+ " REPORTED ERROR: "+error);
    }

    this->prepared_query = cass_future_get_prepared(future);
    cass_future_free(future);


    this->session = session;
    this->table_metadata = table_meta;
    this->myCache = new Poco::LRUCache<TupleRow, TupleRow>(cache_size);
    this->keys_factory = new TupleRowFactory(table_meta->get_keys());
    this->values_factory = new TupleRowFactory(table_meta->get_values());
};


CacheTable::~CacheTable() {
    myCache->clear();// destroys keys
    delete(myCache);
    delete (keys_factory);
    delete (values_factory);
    cass_prepared_free(prepared_query);
    prepared_query=NULL;
    session=NULL;
}


void CacheTable::put_crow(void* keys, void* values) {
    const TupleRow *k = keys_factory->make_tuple(keys);
    const TupleRow *v = values_factory->make_tuple(values);
    this->myCache->update(*k,v); //Inserts if not present, otherwise replaces
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
TupleRow* CacheTable::retrieve_from_cassandra(const TupleRow *keys){

    /* Not present on cache, a query is performed */
    CassStatement *statement = cass_prepared_bind(prepared_query);

    this->keys_factory->bind(statement,keys,0);

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
    const CassRow *row = cass_result_first_row(result);
    TupleRow* tuple_result=values_factory->make_tuple(row);

    cass_result_free(result);
    return tuple_result;
}


const TupleRow *CacheTable::get_crow(const TupleRow *keys) {
    Poco::SharedPtr<TupleRow> ptrElem = myCache->get(*keys);
    if (!ptrElem.isNull()) {
        return ptrElem.get();
    }

    const TupleRow *values = retrieve_from_cassandra(keys);

    myCache->add(*keys, values);
    //Store result to cache
        //TODO it calls TupleRow::TupleRow(const TupleRow *t) for values which is wrong
    return values;
}


std::shared_ptr<void> CacheTable::get_crow(void* keys) {
    return get_crow(keys_factory->make_tuple(keys))->get_payload();
}
