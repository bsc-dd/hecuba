#include "Prefetch.h"


Prefetch::Prefetch(const std::pair<uint64_t, uint64_t> *token_ranges, uint32_t buff_size, CacheTable *cache_table, CassSession *session,
                   const char *query, uint16_t n_ranges) {
    this->cache = cache_table;
    this->session = session;
    this->t_factory = cache_table->get_tuple_f();
    this->tokens=token_ranges;
    this->n_tokens=n_ranges;
    CassFuture *future = cass_session_prepare(session, query);
    CassError rc = cass_future_error_code(future);
    if (rc != CASS_OK) {
        throw new std::runtime_error(cass_error_desc(rc));
    }
    this->prepared_query = cass_future_get_prepared(future);
    this->data.set_capacity(buff_size);

    this->worker = new std::thread{&Prefetch::consume_tokens,this};

}

//unsigned long long tokens_list[][2]
void Prefetch::consume_tokens() {
    for (uint16_t i = 0; i < n_tokens; ++i) {

        CassStatement *statement = cass_prepared_bind(this->prepared_query);
        cass_statement_bind_int64(statement, 0, tokens[i].first);
        cass_statement_bind_int64(statement, 1, tokens[i].second);

        CassFuture *future = cass_session_execute(session, statement);
        cass_statement_free(statement);
        // BLOCKS AND WAITS
        const CassResult *result = cass_future_get_result(future);
        cass_future_free(future);
        CassIterator *iterator = cass_iterator_from_result(result);

        while (cass_iterator_next(iterator)) {
            const CassRow *row = cass_iterator_get_row(iterator);
            TupleRow *t = t_factory->make_tuple(row);
            data.push(t); //blocking operation
            cache->put_row(row); //would be better to pass the tuple
        }
        cass_iterator_free(iterator);
        cass_result_free(result);
    }
    data.push(NULL);
}


















/*
.
TokenGenerator* token_gen = new TokenGenerator();


void storeResult(const CassResult* result) {

    CassIterator* iterator = cass_iterator_from_result(result);
    while (cass_iterator_next(iterator)) {
        const CassRow* row = cass_iterator_get_row(iterator);
        CassIterator *col_it = cass_iterator_from_row(row);
        int ncol = 0;

        field arr_fields[num_fields];

        while (cass_iterator_next(col_it)) {
            cass_value_get_bytes(cass_iterator_get_column(col_it), &arr_fields[ncol].data, &arr_fields[ncol].size);
            ++ncol;
        }
        bytes
                cache[arr_fields[pos_pk].data] = arr_fields;
        cass_iterator_free(col_it);
    }
    cass_iterator_free(iterator);

}



void callback(CassFuture *future, void *ptr) {
    CassError rc = cass_future_error_code(future);
    if (rc != CASS_OK) {
        printf("%s \n", cass_error_desc(rc));
    }
    const CassResult* result = cass_future_get_result(future);

    storeResult(result);

    cass_result_free(result);

    call_async();
}


void call_async() {
    std::vector<unsigned long long> tokens = token_gen->get_next();
    if (tokens.size()==0) {
        return;
    }

    CassStatement *statement = cass_prepared_bind(prepared);
    cass_statement_bind_int64(statement, 0, tokens[0]);
    cass_statement_bind_int64(statement, 1, tokens[1] );
    CassFuture *query_future = cass_session_execute(session, statement);

    cass_statement_free(statement);
    cass_future_set_callback(query_future, callback, NULL);
    cass_future_free(query_future);
}




void call_sync(PyObject *key) {

    CassStatement *statement = cass_prepared_bind(prepared);

//CassError er = cass_statement_bind_bytes(statement,0, key,size);
//if (er!=CASS_OK) printf(" Err %s %d\n", cass_error_desc(er), *key);
    bind_key(statement,key);

    std::cout << "bind done " << std::endl;
    CassFuture *query_future = cass_session_execute(session, statement);

    CassError rc = cass_future_error_code(query_future);
    const CassResult* result = cass_future_get_result(query_future);
    cass_future_free(query_future);


    if (result == NULL) {
        printf("%s %d\n", cass_error_desc(rc), *key);
        //printf("SY %u%u%u%u - %zu \n",key[0],key[1],key[2],key[3],size);
        cass_future_free(query_future);
        return;
    }


    storeResult(result);

    cass_result_free(result);
    cass_statement_free(statement);

}


*/

