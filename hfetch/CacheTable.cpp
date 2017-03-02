#include "CacheTable.h"

#define writer_buff_size 100
#define max_write_callbacks 4

/***
 * Constructs a cache which takes and returns data encapsulated as pointers to TupleRow or PyObject
 * Follows Least Recently Used replacement strategy
 * @param size Max elements the cache will hold, afterwards replacement takes place
 * @param table Name of the table being represented
 * @param keyspace Name of the keyspace whose table belongs to
 * @param query Query ready to be bind with the keys
 * @param session
 */
CacheTable::CacheTable(uint32_t size, const std::string &table, const std::string &keyspace,
const std::vector<std::string> &keyn,
const std::vector< std::vector<std::string>>  &columns_n ,
const std::string &token_range_pred,
const std::vector<std::pair<int64_t, int64_t>> &tkns,
        CassSession *session) {
    columns_names = columns_n;
    key_names = keyn;
    tokens=tkns;
    token_predicate = "FROM " + keyspace + "." + table + " " + token_range_pred;
    get_predicate = "FROM " + keyspace + "." + table + " WHERE " + key_names[0] + "=?";
    select_keys = "SELECT " + key_names[0];
    for (uint16_t i = 1; i < key_names.size(); i++) {
        get_predicate += " AND " + key_names[i] + "=?";
        select_keys += "," + key_names[i];
    }

    select_values = columns_names[0][0];
    for (uint16_t i = 1; i < columns_names.size(); i++) {
        select_values += "," + columns_names[i][0];
    }
    select_values+=" ";
    select_keys+=" ";

    select_all = select_keys + "," + select_values;

    select_values = "SELECT " + select_values;

    this->myCache = new Poco::LRUCache<TupleRow, TupleRow>(size);
    cache_query = select_values + get_predicate;
    CassFuture *future = cass_session_prepare(session, cache_query.c_str());
    CassError rc = cass_future_error_code(future);
    if (rc!=CASS_OK) {
        std::string error = cass_error_desc(rc);
        throw ModuleException("Cache particles_table: Preparing query: "+cache_query+ " REPORTED ERROR: "+error);
    }

    prepared_query = cass_future_get_prepared(future);

    this->session = session;
    cass_future_free(future);
    //lacks free future (future)
    const CassSchemaMeta *schema_meta = cass_session_get_schema_meta(session);
    if (!schema_meta) {
        throw ModuleException("Cache particles_table: constructor: Schema meta is NULL");
    }

    const CassKeyspaceMeta *keyspace_meta = cass_schema_meta_keyspace_by_name(schema_meta, keyspace.c_str());
    if (!keyspace_meta) {
        throw ModuleException("Keyspace particles_table: constructor: Schema meta is NULL");
    }


    const CassTableMeta *table_meta = cass_keyspace_meta_table_by_name(keyspace_meta, table.c_str());
    if (!table_meta || (cass_table_meta_column_count(table_meta)==0)) {
        throw ModuleException("Cache particles_table: constructor: Table meta is NULL");
    }

    std::vector < std::vector< std::string > > keys_copy(key_names.size(),std::vector< std::string>(1));

    for (int i = 0; i<key_names.size(); ++i) {
        keys_copy[i][0]=key_names[i];
    }

    all_names.reserve(key_names.size() + columns_names.size());
    all_names.insert(all_names.end(), keys_copy.begin(), keys_copy.end());
    all_names.insert(all_names.end(), columns_names.begin(), columns_names.end());
    try {
    keys_factory = new TupleRowFactory(table_meta, keys_copy);
    values_factory = new TupleRowFactory(table_meta, columns_names);
    items_factory = new TupleRowFactory(table_meta, all_names);
    }
    catch (ModuleException e) {
        throw e;
    }
    cass_schema_meta_free(schema_meta);
    std::string write_query = "INSERT INTO "+keyspace+"."+table+"(";

    write_query+=all_names[0][0];
    for (uint16_t i = 1;i<all_names.size(); ++i) {
        write_query+=","+all_names[i][0];
    }
    write_query+=") VALUES (?";
    for (uint16_t i = 1; i<all_names.size();++i) {
        write_query+=",?";
    }
    write_query+=");";
    writer = new Writer((uint16_t )writer_buff_size,(uint16_t )max_write_callbacks,*keys_factory,*values_factory,session,write_query);
};


CacheTable::~CacheTable() {
    cass_prepared_free(prepared_query);
    //stl tree calls deallocate for cache nodes on clear()->erase(), and later on destroy, which ends up calling the deleters
    delete(writer); //First of all, needs to flush the data using the key and values factory
    myCache->clear();// destroys keys
    delete(myCache);
    delete (keys_factory);
    delete (values_factory);
    delete (items_factory);
    prepared_query=NULL;
    session=NULL;
}




Prefetch* CacheTable::get_keys_iter(uint32_t prefetch_size) {
    return new Prefetch(tokens, prefetch_size, *keys_factory, session, select_keys + token_predicate);
}

Prefetch* CacheTable::get_values_iter(uint32_t prefetch_size) {
    return new Prefetch(tokens, prefetch_size, *values_factory, session, select_values + token_predicate);
}

Prefetch* CacheTable::get_items_iter(uint32_t prefetch_size) {
    return new Prefetch(tokens, prefetch_size, *items_factory, session, select_all + token_predicate);
}




/***
 * This method converts a python object (list) and converts it to a C++ TupleRow
 * and inserts it into the Cache.
 * The object is then put in a queue for being inserted into Cassandra.
 * @param row
 * @return
 */

void CacheTable::put_row(PyObject *key, PyObject *value) {
    TupleRow *k = keys_factory->make_tuple(key);
    const TupleRow *v = values_factory->make_tuple(value);
    //Inserts if not present, otherwise replaces
    this->myCache->update(*k, v);
    this->writer->write_to_cassandra(k,v);
}


PyObject *CacheTable::get_row(PyObject *py_keys) {

    TupleRow *keys = keys_factory->make_tuple(py_keys);
    const TupleRow *values = get_crow(keys);
    delete(keys);

    if(values==NULL){
        std::cout << "CacheTable: Get Row: VALUES IS NULL " << std::endl;
        PyErr_SetString(PyExc_KeyError,"Get row: key not found");
        return NULL;
    }

    PyObject* temp = values_factory->tuple_as_py(values);
    return temp;
}



const TupleRow *CacheTable::get_crow(TupleRow *keys) {

    Poco::SharedPtr<TupleRow> ptrElem = myCache->get(*keys);
    if (!ptrElem.isNull()) {
        return ptrElem.get();
    }
    /* Not present on cache, a query is performed */
    CassStatement *statement = cass_prepared_bind(prepared_query);

    this->keys_factory->bind(statement,keys,0);

    CassFuture *query_future = cass_session_execute(session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    CassError rc = cass_future_error_code(query_future);
    if (result == NULL) {
        /* Handle error */
        printf("%s\n", cass_error_desc(rc));
        cass_future_free(query_future);
        cass_statement_free(statement);
        return NULL;
    }

    cass_future_free(query_future);
    cass_statement_free(statement);
    if (0 == cass_result_row_count(result)) {
        return NULL;
    }

    const CassRow *row = cass_result_first_row(result);

    //Store result to cache
    const TupleRow *values = values_factory->make_tuple(row);

    myCache->add(*keys, values);
    cass_result_free(result);
    return values;
}