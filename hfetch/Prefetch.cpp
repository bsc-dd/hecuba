#include "Prefetch.h"

#define MAX_TRIES 10

Prefetch::Prefetch(const std::vector<std::pair<int64_t, int64_t>> &token_ranges, const TableMetadata* table_meta,
                   CassSession* session,uint32_t prefetch_size) {
    if (!session)
        throw ModuleException("Prefetch: Session is Null");
    this->session = session;
    this->t_factory = TupleRowFactory(table_meta->get_items());
    this->tokens = token_ranges;
    this->completed = false;
    this->error_msg = NULL;
    CassFuture *future = cass_session_prepare(session, table_meta->get_select_tokens());
    CassError rc = cass_future_error_code(future);
    CHECK_CASS("prefetch cannot prepare"); //TODO when prepare doesnt succeed, crashes later with segfault
    this->prepared_query = cass_future_get_prepared(future);
    cass_future_free(future);
    this->data.set_capacity(prefetch_size);
    this->worker = new std::thread{&Prefetch::consume_tokens, this};

}

Prefetch::~Prefetch() {
    data.set_capacity(0);

    while (!completed) data.abort();

    worker->join();
    delete (worker);

    TupleRow *to_delete;
    while (data.try_pop(to_delete)) delete (to_delete);

    if (this->prepared_query != NULL) cass_prepared_free(this->prepared_query);
}



TupleRow *Prefetch::get_cnext() {
    if (completed&&data.empty()) return NULL;
    TupleRow *response;
    try {
        data.pop(response);
    }
    catch (std::exception &e) {
        if (data.empty()) return NULL;
        else return get_cnext();
    }
    return response;
}


void Prefetch::consume_tokens() {
    for (std::pair<int64_t, int64_t> &range : tokens) {
        //If Consumer sets capacity 0, we stop fetching data
        if (data.capacity() == 0) {
            completed = true;
            data.abort();
            return;
        }

        //Bind tokens and execute
        CassStatement *statement = cass_prepared_bind(this->prepared_query);
        cass_statement_bind_int64(statement, 0, range.first);
        cass_statement_bind_int64(statement, 1, range.second);
        CassFuture *future = cass_session_execute(session, statement);
        cass_statement_free(statement);

        const CassResult *result = NULL;
        int tries = 0;

        while (result == NULL) {
            //If Consumer sets capacity 0, we stop fetching data
            if (data.capacity() == 0) {
                cass_future_free(future);
                completed=true;
                data.abort();
                return;
            }

            result = cass_future_get_result(future);
            CassError rc = cass_future_error_code(future);

            if (rc != CASS_OK) {
                std::cerr << "Prefetch action failed: " << cass_error_desc(rc) << " Try #" << tries << std::endl;
                tries++;
                if (tries > MAX_TRIES) {
                    cass_future_free(future);
                    completed = true;
                    data.abort();
                    std::cerr << "Prefetch reached max connection attempts " << MAX_TRIES << std::endl;
                    return;
                }
            }
        }


        //PRE: Result != NULL, future != NULL, completed = false
        cass_future_free(future);

        CassIterator *iterator = cass_iterator_from_result(result);
        while (cass_iterator_next(iterator)) {
            if (data.capacity() == 0) {
                completed = true;
                data.abort();
                cass_iterator_free(iterator);
                cass_result_free(result);
                return;
            }
            const CassRow *row = cass_iterator_get_row(iterator);
            TupleRow *t = t_factory.make_tuple(row);
            try {
                data.push(t); //blocking operation
            }
            catch (std::exception &e) {
                completed = true;
                data.abort();
                delete (t);
                cass_iterator_free(iterator);
                cass_result_free(result);
                return;
            }
        }
        //Done fetching current token range
        cass_iterator_free(iterator);
        cass_result_free(result);
    }
    //All token ranges fetched
    completed = true;
    data.abort();
}