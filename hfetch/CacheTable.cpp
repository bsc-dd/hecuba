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
    k->get_payload().reset();
    v->get_payload().reset((void*)NULL,[](void *ptr) {});
    delete(k);
    delete(v);
}


/** this method only adds the data to the cache
 *  without makingit persistent
 * @param keys
 * @param values
 */

void CacheTable::add_to_cache(void* keys, void* values) {
    const TupleRow *k = keys_factory->make_tuple(keys);
    const TupleRow *v = values_factory->make_tuple(values);
    if (myCache) this->myCache->update(*k,v);
    k->get_payload().reset();
    v->get_payload().reset((void*)NULL,[](void *ptr) {});
    delete(k);
    delete(v);
}


void CacheTable::put_crow(const TupleRow* row) {
    //split into two and call put_crow(a,b);
    std::shared_ptr<const std::vector<ColumnMeta> > keys_meta = keys_factory->get_metadata();
    uint16_t nkeys = (uint16_t)keys_meta->size();
    std::shared_ptr<const std::vector<ColumnMeta> > values_meta = values_factory->get_metadata();
    uint16_t nvalues = (uint16_t)values_meta->size();

    char* keys = (char*) malloc(keys_meta->at(nkeys-(uint16_t)1).position+keys_meta->at(nkeys-(uint16_t)1).size);
    char* values = (char*) malloc(values_meta->at(nvalues-(uint16_t)1).position+values_meta->at(nvalues-(uint16_t)1).size);
    TupleRow *k = keys_factory->make_tuple(keys);
    TupleRow *v = values_factory->make_tuple(values);
    //Copy keys
    for (uint16_t i=0; i<nkeys; ++i) {
        CassValueType type = keys_meta->at(i).type;
        const void *element_i = row->get_element(i);
        if (element_i != nullptr) {
            if (type == CASS_VALUE_TYPE_BLOB) {
                char **from = (char **) element_i;
                char *from_data = *from;

                uint64_t *size = (uint64_t *) from_data;

                void *new_data = malloc(*size);
                memcpy(new_data, from_data, *size + sizeof(uint64_t));
                //Copy ptr
                memcpy(keys + keys_meta->at(i).position, &new_data, keys_meta->at(i).size);
            } else if (type == CASS_VALUE_TYPE_TEXT || type == CASS_VALUE_TYPE_VARCHAR ||
                       type == CASS_VALUE_TYPE_ASCII) {

                char **from = (char **) element_i;
                char *from_data = *from;

                uint64_t size = strlen(from_data);

                void *new_data = malloc(size);
                memcpy(new_data, from_data, size);
                //Copy ptr
                memcpy(keys + keys_meta->at(i).position, &new_data, keys_meta->at(i).size);
            } else if (type == CASS_VALUE_TYPE_UUID) {

                uint64_t **from = (uint64_t **) element_i;

                uint64_t size = sizeof(uint64_t) * 2;
                void *new_data = malloc(size);
                memcpy(new_data, *from, size);
                //Copy ptr
                memcpy(keys + keys_meta->at(i).position, &new_data, keys_meta->at(i).size);
            }
            else memcpy(keys + keys_meta->at(i).position, element_i, keys_meta->at(i).size);
        }
        else  k->setNull(i);
    }


    //Copy values
    for (uint16_t i=0; i<nvalues; ++i) {

        CassValueType type = values_meta->at(i).type;
        const void *element_i = row->get_element(i+nkeys);
        if (element_i != nullptr) {
        if (type==CASS_VALUE_TYPE_BLOB) {
            char **from = (char**) element_i;
            char *from_data = *from;

            uint64_t *size = (uint64_t*)from_data;

            void *new_data = malloc(*size);
            memcpy(new_data,from_data,*size+sizeof(uint64_t));
            //Copy ptr
            memcpy(values+values_meta->at(i).position,&new_data,values_meta->at(i).size);
        } else if (type==CASS_VALUE_TYPE_TEXT || type==CASS_VALUE_TYPE_VARCHAR || type == CASS_VALUE_TYPE_ASCII) {

            char **from = (char**) element_i;
            char *from_data = *from;

            uint64_t size = strlen(from_data);

            void *new_data = malloc(size);
            memcpy(new_data,from_data,size);
            //Copy ptr
            memcpy(values+values_meta->at(i).position,&new_data,values_meta->at(i).size);
        }
        else if (type==CASS_VALUE_TYPE_UUID){

            uint64_t **from = (uint64_t**)element_i;

            uint64_t size = sizeof(uint64_t)*2;
            void *new_data = malloc(size);
            memcpy(new_data,*from,size);
            //Copy ptr
            memcpy(values+values_meta->at(i).position,&new_data,values_meta->at(i).size);
        }
        else memcpy(values+values_meta->at(i).position,element_i,values_meta->at(i).size);
        }
        else v->setNull(i);
    }
    this->add_to_cache(keys,values);
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


std::vector<const TupleRow *>  CacheTable::get_crow(const TupleRow *keys) {
    if (myCache) {
        Poco::SharedPtr<TupleRow> ptrElem = myCache->get(*keys);
        if (!ptrElem.isNull())  return std::vector <const TupleRow*> { new TupleRow(ptrElem.get()) };
    }

    std::vector<const TupleRow *> values = retrieve_from_cassandra(keys);

    if (myCache && !values.empty()) myCache->add(*keys, values[0]);

    return values;
}


std::vector<std::shared_ptr<void>> CacheTable::get_crow(void* keys) {
    const TupleRow* tuple_key = keys_factory->make_tuple(keys);
    std::vector<const TupleRow*> result = get_crow(tuple_key);
    delete(tuple_key);
    
    if (result.empty()) return std::vector<std::shared_ptr<void> >(0);
    
    std::vector<std::shared_ptr<void>> payloads(result.size());
    for (uint32_t i = 0; i<result.size(); ++i) {
        payloads[i]=result[i]->get_payload();
        delete(result[i]);
    }
    return payloads;
}
