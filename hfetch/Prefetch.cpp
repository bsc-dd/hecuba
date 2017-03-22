#include "Prefetch.h"

#define MAX_TRIES 10

Prefetch::Prefetch(const std::vector<std::pair<int64_t, int64_t>> &token_ranges, uint32_t buff_size,
                   TupleRowFactory& tuple_factory, CassSession *session, std::string query) {
    this->session = session;
    this->t_factory = tuple_factory;
    this->tokens = token_ranges;
    this->completed = false;
    this->error_msg = NULL;
    CassFuture *future = cass_session_prepare(session, query.c_str());
    CassError rc = cass_future_error_code(future);
    CHECK_CASS("prefetch cannot prepare");
    this->prepared_query = cass_future_get_prepared(future);
    cass_future_free(future);
    this->data.set_capacity(buff_size);
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


PyObject *Prefetch::get_next() {
    const TupleRow *response = get_cnext();
    if (response == NULL) {
        if (error_msg == NULL) {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        } else {
            PyErr_SetString(PyExc_RuntimeError, error_msg);
            return NULL;
        }
    }
    std::vector<const TupleRow*> row = {response};
    PyObject *toberet = t_factory.tuples_as_py(row);
    delete (response);
    return toberet;
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