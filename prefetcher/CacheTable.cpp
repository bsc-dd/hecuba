#include "CacheTable.h"


CacheTable::CacheTable(uint32_t size, const char * table, const char * keyspace, const char* query, CassSession* session) {


    this->myCache=new Poco::LRUCache<TupleRow,TupleRow>(size);

    CassFuture* future=cass_session_prepare(session,query);
    CassError   rc = cass_future_error_code(future);
    if(rc!=CASS_OK){
        throw new std::runtime_error(cass_error_desc(rc));
    }
    prepared_query = cass_future_get_prepared(future);

    this->session=session;
    //lacks free future (future)
    const CassSchemaMeta *schema_meta = cass_session_get_schema_meta(session);
    assert(schema_meta != NULL &&  "error on schema");
    const CassKeyspaceMeta *keyspace_meta = cass_schema_meta_keyspace_by_name(schema_meta, keyspace);

    assert(keyspace_meta != NULL && "error on keyspace");
    const CassTableMeta *table_meta = cass_keyspace_meta_table_by_name(keyspace_meta,table);
    t_factory = new TupleRowFactory(table_meta);
    cass_schema_meta_free(schema_meta);
};





CacheTable::~CacheTable() {
    cass_prepared_free(prepared_query);
    //stl tree calls deallocate for cache nodes on clear()->erase(), and later on destroy, which ends up calling the deleters
    myCache->clear();
    delete(t_factory);
}



TupleRowFactory* CacheTable::get_tuple_f() {
    return this->t_factory;
}



void CacheTable::bind_keys(CassStatement *statement, TupleRow* keys) {

    for (uint16_t i = 0; i < keys->size(); ++i) {
        const void *key = keys->get_element(i);
        uint16_t bind_pos = i;
        switch (t_factory->get_key_type(i)) {
            case CASS_VALUE_TYPE_VARCHAR:
            case CASS_VALUE_TYPE_TEXT:
            case CASS_VALUE_TYPE_ASCII: {
                const char *temp = static_cast<const char *>(key);
                cass_statement_bind_string(statement, bind_pos, temp);
                break;
            }
            case CASS_VALUE_TYPE_VARINT:
            case CASS_VALUE_TYPE_BIGINT: {
                const int64_t *data = static_cast<const int64_t *>(key);
                cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                break;
            }
            case CASS_VALUE_TYPE_BLOB: {
                //cass_statement_bind_bytes(statement,bind_pos,key,size);
                break;
            }
            case CASS_VALUE_TYPE_BOOLEAN: {
                cass_bool_t b = cass_false;
                const bool *bindbool = static_cast<const bool *>(key);
                if (*bindbool) b = cass_true;
                cass_statement_bind_bool(statement, bind_pos, b);
                break;
            }
            case CASS_VALUE_TYPE_COUNTER: {
                const uint64_t *data = static_cast<const uint64_t *>(key);
                cass_statement_bind_int64(statement, bind_pos, *data);//L means long long, K unsigned long long
                break;
            }
            case CASS_VALUE_TYPE_DECIMAL: {
                //decimal.Decimal
                break;
            }
            case CASS_VALUE_TYPE_DOUBLE: {
                const double *data = static_cast<const double *>(key);
                cass_statement_bind_double(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_FLOAT: {
                const float *data = static_cast<const float *>(key);
                cass_statement_bind_float(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_INT: {
                const int32_t *data = static_cast<const int32_t *>(key);
                cass_statement_bind_int32(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_TIMESTAMP: {

                break;
            }
            case CASS_VALUE_TYPE_UUID: {

                break;
            }
            case CASS_VALUE_TYPE_TIMEUUID: {

                break;
            }
            case CASS_VALUE_TYPE_INET: {

                break;
            }
            case CASS_VALUE_TYPE_DATE: {

                break;
            }
            case CASS_VALUE_TYPE_TIME: {

                break;
            }
            case CASS_VALUE_TYPE_SMALL_INT: {
                const int16_t *data = static_cast<const int16_t *>(key);
                cass_statement_bind_int16(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_TINY_INT: {
                const int8_t *data = static_cast<const int8_t *>(key);
                cass_statement_bind_int8(statement, bind_pos, *data);
                break;
            }
            case CASS_VALUE_TYPE_LIST: {
                break;
            }
            case CASS_VALUE_TYPE_MAP: {

                break;
            }
            case CASS_VALUE_TYPE_SET: {

                break;
            }
            case CASS_VALUE_TYPE_TUPLE: {
                break;
            }
            default://CASS_VALUE_TYPE_UDT|CASS_VALUE_TYPE_CUSTOM|CASS_VALUE_TYPE_UNKNOWN:
                break;
        }
    }

}


/***
 * This method converts a python object (list) and converts it to a C++ TupleRow
 * and inserts it into the Cache.
 * The object is then put in a queue for being inserted into Cassandra.
 * @param row
 * @return
 */

int CacheTable::put_row(PyObject *row) {
    TupleRow* t = t_factory->make_tuple(row);
    TupleRow* keys = t_factory->extract_key_tuple(row);
    //Inserts if not present, otherwise replaces
    myCache->update(*keys,t);
    return 0;
}


int CacheTable::put_row(const CassRow *row) {
    TupleRow* t = t_factory->make_tuple(row);
    TupleRow* keys = t_factory->extract_key_tuple(row);
    //Inserts if not present, otherwise replaces
    myCache->update(*keys,t);
    return 0;
}


/***
 * This method  does:
 * 1. converts the python object to C
 * 2. check if it is present in cache. If yes, returns it
 * 3. otherwise, queries Cassandra
 * 4. If a value is returned, it is put into the Cache, converted to Python and returned to the user.
 *
 * //TODO to be moved somewhere else :)
 * @param py_keys
 * @return
 */
//#define bm
PyObject *CacheTable::get_row(PyObject *py_keys) {

    TupleRow *keys;
#ifndef bm
    keys = t_factory->make_key_tuple(py_keys);

    Poco::SharedPtr<TupleRow> ptrElem = myCache->get(*keys);
    if (!ptrElem.isNull()) {

        return t_factory->tuple_as_py(ptrElem.get());
    }
#endif

#ifdef bm
    Py_RETURN_TRUE;
#endif
    /* Not present on cache, a query is performed */
    CassStatement *statement = cass_prepared_bind(prepared_query);

    bind_keys(statement, keys);

    CassFuture *query_future = cass_session_execute(session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    CassError rc = cass_future_error_code(query_future);
    if (result == NULL) {
        /* Handle error */
        printf("%s\n", cass_error_desc(rc));
        cass_future_free(query_future);
        return NULL;
    }

    cass_future_free(query_future);
    cass_statement_free(statement);
    if (0==cass_result_row_count(result)) //or retry 2 times
        return NULL;
    const CassRow* row = cass_result_first_row(result);

    //Store result to cache
    TupleRow* values = t_factory->make_tuple(row);

    myCache->add(*keys,values);
    cass_result_free(result);
    return t_factory->tuple_as_py(values);
}


//https://github.com/pocoproject/poco/blob/develop/Foundation/include/Poco/AbstractCache.h#L218
//Cache allocates space for each insert



/* *********** DEBUG PURPOSES **************/
/*
int temp = 123;
TupleRow *CacheTable::get_row_debug(int pkey) {
    temp=pkey;
    TupleRow *keys = new TupleRow(1,&elem_sizes);
    keys->set_element(&temp,0);

    Poco::SharedPtr<TupleRow*> ptrElem = myCache.get(*keys);
    if (!ptrElem.isNull()) {
        std::cout << "PRESENT" << std::endl;
        return *ptrElem.get_row();
    }
    CassStatement *statement = cass_statement_new("SELECT * FROM particle WHERE partid = 123 LIMIT 1;", 0);

//    cass_statement_bind_int32(statement,temp,0);
    CassFuture *query_future = cass_session_execute(session, statement);

    const CassResult *result = cass_future_get_result(query_future);
    CassError rc = cass_future_error_code(query_future);
    if (result == NULL) {
        printf("%s\n", cass_error_desc(rc));
        return NULL;
    }
    cass_future_free(query_future);

    cass_statement_free(statement);
    const CassRow* row = cass_result_first_row(result);


    TupleRow* resp = insert_row(row);

    cass_result_free(result);
    return resp; //result lost in memory
}
*/