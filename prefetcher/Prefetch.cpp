#include "Prefetch.h"


#define CHECK_CASS(msg) if(rc != CASS_OK){ \
std::cerr<<msg<<std::endl; };\
//throw ModuleException(msg); };

Prefetch::Prefetch(const std::vector<std::pair<int64_t, int64_t>> *token_ranges, uint32_t buff_size,
                   TupleRowFactory *tuple_factory, CassSession *session, std::string query) {
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

PyObject *Prefetch::get_next() {
    if (completed) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    TupleRow *response = NULL;
    data.pop(response);
    if (!response || response->n_elem() == 0) {
        completed = true;
        if (error_msg == NULL) {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        } else {
            PyErr_SetString(PyExc_RuntimeError, error_msg);
            return NULL;

        }
    }

    PyObject *toberet = t_factory->tuple_as_py(response);
    delete (response);

    return toberet;
}

void Prefetch::consume_tokens() {
    for (std::pair<int64_t, int64_t> range : *tokens) {
        CassStatement *statement = cass_prepared_bind(this->prepared_query);
        cass_statement_bind_int64(statement, 0, range.first);
        cass_statement_bind_int64(statement, 1, range.second);

        CassFuture *future = cass_session_execute(session, statement);
        cass_statement_free(statement);
        const CassResult *result = NULL;
        int tries = 0;
        while (result == NULL && tries < 10) {
            result = cass_future_get_result(future);
            cass_future_free(future);
            if (result == NULL) {
                CassError rc = cass_future_error_code(future);
                std::cout << cass_error_desc(rc) << std::endl;
                tries++;

            } else {
                CassIterator *iterator = cass_iterator_from_result(result);

                while (cass_iterator_next(iterator)) {
                    const CassRow *row = cass_iterator_get_row(iterator);
                    TupleRow *t = t_factory->make_tuple(row);
                    try {
                        data.push(t); //blocking operation
                    }
                    catch (std::exception &e) {
                        std::cerr << "killing the thread" << std::endl;
                        delete (t);
                        cass_iterator_free(iterator);
                        cass_result_free(result);
                        return;

                    }
                }
                cass_iterator_free(iterator);
                cass_result_free(result);
            }
        }
        try {
            if (tries == 10)
                this->error_msg = "impossible to get data";
            data.push(NULL);
        }
        catch (tbb::user_abort &e) {

            return;

        }
    }

}