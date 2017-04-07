#include "CacheTable.h"

#define default_writer_buff 100
#define default_writer_callbacks 4
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
CacheTable::CacheTable(const std::string &table, const std::string &keyspace,
                       const std::vector<std::string> &keyn,
                       const std::vector<std::vector<std::string>> &columns_n,
                       const std::string &token_range_pred,
                       const std::vector<std::pair<int64_t, int64_t>> &tkns,
                       CassSession *session,
                       std::map<std::string, std::string> &config) {


    int32_t writer_num_callbacks = default_writer_callbacks;
    int32_t writer_buffer_size = default_writer_buff;
    int32_t cache_size = default_cache_size;

    if (config.find("writer_par") != config.end()) {
        std::string wr_calls = config["writer_par"];
        try {
            writer_num_callbacks = std::stoi(wr_calls);
            if (writer_num_callbacks < 0) throw ModuleException("Writer parallelism value must be >= 0");
        }
        catch (std::exception e) {
            std::string msg(e.what());
            msg += " Malformed value in config for writer_par";
            throw ModuleException(msg);
        }
    }

    if (config.find("writer_buffer") != config.end()) {
        std::string wr_buff = config["writer_buffer"];
        try {
            writer_buffer_size = std::stoi(wr_buff);
            if (writer_buffer_size < 0) throw ModuleException("Writer buffer value must be >= 0");
        }
        catch (std::exception e) {
            std::string msg(e.what());
            msg += " Malformed value in config for writer_buffer";
            throw ModuleException(msg);
        }
    }

    if (config.find("cache_size") != config.end()) {
        std::string cache_size_str = config["cache_size"];
        try {
            cache_size = std::stoi(cache_size_str);
            if (cache_size < 0) throw ModuleException("Cache size value must be >= 0");
        }
        catch (std::exception e) {
            std::string msg(e.what());
            msg += " Malformed value in config for cache_size";
            throw ModuleException(msg);
        }
    }

    this->keyspace = keyspace;
    /** Parse names **/

    columns_names = columns_n;
    key_names = keyn;
    tokens = tkns;
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
    select_values += " ";
    select_keys += " ";

    select_all = select_keys + "," + select_values;

    select_values = "SELECT " + select_values;

    this->myCache = new Poco::LRUCache<TupleRow, TupleRow>(cache_size);
    cache_query = select_values + get_predicate;
    CassFuture *future = cass_session_prepare(session, cache_query.c_str());
    CassError rc = cass_future_error_code(future);
    if (rc != CASS_OK) {
        std::string error = cass_error_desc(rc);
        throw ModuleException("Cache particles_table: Preparing query: " + cache_query + " REPORTED ERROR: " + error);
    }

    prepared_query = cass_future_get_prepared(future);

    this->session = session;
    cass_future_free(future);

    const CassSchemaMeta *schema_meta = cass_session_get_schema_meta(session);
    if (!schema_meta) {
        throw ModuleException("Cache particles_table: constructor: Schema meta is NULL");
    }

    const CassKeyspaceMeta *keyspace_meta = cass_schema_meta_keyspace_by_name(schema_meta, keyspace.c_str());
    if (!keyspace_meta) {
        throw ModuleException("Keyspace particles_table: constructor: Schema meta is NULL");
    }


    const CassTableMeta *table_meta = cass_keyspace_meta_table_by_name(keyspace_meta, table.c_str());
    if (!table_meta || (cass_table_meta_column_count(table_meta) == 0)) {
        throw ModuleException("Cache particles_table: constructor: Table meta is NULL");
    }

    std::vector<std::vector<std::string> > keys_copy(key_names.size(), std::vector<std::string>(1));

    for (uint16_t i = 0; i < key_names.size(); ++i) {
        keys_copy[i][0] = key_names[i];
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
    std::string write_query = "INSERT INTO " + keyspace + "." + table + "(";

    //this can be done in one for loop TODO
    write_query += all_names[0][0];
    for (uint16_t i = 1; i < all_names.size(); ++i) {
        write_query += "," + all_names[i][0];
    }
    write_query += ") VALUES (?";
    for (uint16_t i = 1; i < all_names.size(); ++i) {
        write_query += ",?";
    }
    write_query += ");";

    writer = new Writer((uint16_t) writer_buffer_size, (uint16_t) writer_num_callbacks, *keys_factory, *values_factory,
                        session, write_query);
};


CacheTable::~CacheTable() {
    cass_prepared_free(prepared_query);
    //stl tree calls deallocate for cache nodes on clear()->erase(), and later on destroy, which ends up calling the deleters
    delete (writer); //First of all, needs to flush the data using the key and values factory
    myCache->clear();// destroys keys
    delete (myCache);
    delete (keys_factory);
    delete (values_factory);
    delete (items_factory);
    prepared_query = NULL;
    session = NULL;
}


Prefetch *CacheTable::get_keys_iter(uint32_t prefetch_size) {
    return new Prefetch(tokens, prefetch_size, *keys_factory, session, select_keys + token_predicate);
}

Prefetch *CacheTable::get_values_iter(uint32_t prefetch_size) {
    return new Prefetch(tokens, prefetch_size, *values_factory, session, select_values + token_predicate);
}

Prefetch *CacheTable::get_items_iter(uint32_t prefetch_size) {
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
    if (!values_factory->get_metadata().has_numpy) {
        TupleRow *k = keys_factory->make_tuple(key);
        const TupleRow *v = values_factory->make_tuple(value);
        //Inserts if not present, otherwise replaces
        this->myCache->update(*k, v);
        this->writer->write_to_cassandra(k, v);
    } else {

        RowMetadata metadata = this->values_factory->get_metadata();
        uint16_t numpy_pos = 0;
        while (metadata.at(numpy_pos).get_arr_type() == NPY_NOTYPE && numpy_pos < metadata.size()) ++numpy_pos;
        if (numpy_pos == metadata.size())
            throw ModuleException("Sth went wrong looking for the numpy");


        if (metadata.at(numpy_pos).info.size() == 5) {
            //else if has auxiliary table
            CassUuid uuid;
            CassUuidGen *uuid_gen = cass_uuid_gen_new();
            cass_uuid_gen_random(uuid_gen, &uuid);
            cass_uuid_gen_free(uuid_gen);


//TODO this awful code will be beautiful when merged with split-c++ branch

            const CassSchemaMeta *schema_meta = cass_session_get_schema_meta(session);
            if (!schema_meta) {
                throw ModuleException("Cache particles_table: constructor: Schema meta is NULL");
            }

            const CassKeyspaceMeta *keyspace_meta = cass_schema_meta_keyspace_by_name(schema_meta, keyspace.c_str());
            if (!keyspace_meta) {
                throw ModuleException("Keyspace particles_table: constructor: Schema meta is NULL");
            }

            std::string table = metadata.at(numpy_pos).info[4];

            const CassTableMeta *table_meta = cass_keyspace_meta_table_by_name(keyspace_meta, table.c_str());
            if (!table_meta || (cass_table_meta_column_count(table_meta) == 0)) {
                throw ModuleException("Cache particles_table: constructor: Table meta is NULL");
            }

            std::vector<std::vector<std::string> > numpy_keys{std::vector<std::string>(1)};
            numpy_keys[0][0] = "uuid";

            std::vector<std::vector<std::string> > numpy_columns{2};
            numpy_columns[0] = {"data", metadata.at(numpy_pos).info[1], metadata.at(numpy_pos).info[2],
                                metadata.at(numpy_pos).info[3]};
            numpy_columns[1] = {"position"};


            Writer *temp = NULL;
            TupleRowFactory npy_keys_f = TupleRowFactory(table_meta, numpy_keys);
            TupleRowFactory npy_values_f = TupleRowFactory(table_meta, numpy_columns);

            cass_schema_meta_free(schema_meta);

            try {
                temp = new Writer(default_writer_buff, default_writer_callbacks, npy_keys_f, npy_values_f, session,
                                  "INSERT INTO " + keyspace + "." + table + " (uuid,data,position) VALUES (?,?,?);");
            } catch (ModuleException e) {
                throw e;
            }

            PyObject *npy_list = PyList_New(1);
            PyObject *array = PyList_GetItem(value, numpy_pos);


            PyList_SetItem(npy_list, 0, array);


            std::vector<const TupleRow *> value_list = npy_values_f.make_tuples_with_npy(npy_list);


            uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);
            *c_uuid = uuid.time_and_version;
            *(c_uuid + 1) = uuid.clock_seq_and_node;

            void *payload = malloc(sizeof(uint64_t *));
            memcpy(payload, &c_uuid, sizeof(uint64_t *));

            TupleRow *numpy_key = new TupleRow(npy_keys_f.get_metadata(), sizeof(uint64_t) * 2, payload);

            for (const TupleRow *T:value_list) {
                TupleRow *key_copy = new TupleRow(numpy_key);
                temp->write_to_cassandra(key_copy, T);
            }

            delete (temp);
            delete (numpy_key);

            //keep numpy key
            PyObject *py_uuid = PyByteArray_FromStringAndSize((char *) c_uuid, sizeof(uint64_t) * 2);


            PyList_SetItem(value, numpy_pos, py_uuid);


            //free numpy key

            TupleRow *k = keys_factory->make_tuple(key);
            const TupleRow *v = values_factory->make_tuple(value);
            //Inserts if not present, otherwise replaces
            this->myCache->update(*k, v);
            this->writer->write_to_cassandra(k, v);
        } else {
            TupleRow *k = keys_factory->make_tuple(key);

            std::vector<const TupleRow *> value_list = values_factory->make_tuples_with_npy(value);
            //this->myCache->update(*k, value_list[0]); <- broken
            for (const TupleRow *T:value_list) {
                TupleRow *key_copy = new TupleRow(k);
                this->writer->write_to_cassandra(key_copy, T);
            }
        }
    }

}


PyObject *CacheTable::get_row(PyObject *py_keys) {


    TupleRow *keys = keys_factory->make_tuple(py_keys);
    std::vector<const TupleRow *> values = get_crow(keys);


    if (values.empty() || values[0] == NULL) {
        delete (keys);
        char *error = (char *) malloc(strlen("Get row: key not found") + 1);
        PyErr_SetString(PyExc_KeyError, error);
        return NULL;
    }

    RowMetadata metadata = values_factory->get_metadata();

    PyObject *row;
    if (metadata.has_numpy) {
        //find numpy pos
        for (uint16_t pos = 0; pos < metadata.size(); ++pos) {
            if (metadata.at(pos).info.size() == 5) {
                //is external

//TODO this awful code will be beautiful when merged with split-c++ branch

                const CassSchemaMeta *schema_meta = cass_session_get_schema_meta(session);
                if (!schema_meta) {
                    throw ModuleException("Cache particles_table: constructor: Schema meta is NULL");
                }

                const CassKeyspaceMeta *keyspace_meta = cass_schema_meta_keyspace_by_name(schema_meta,
                                                                                          keyspace.c_str());
                if (!keyspace_meta) {
                    throw ModuleException("Keyspace particles_table: constructor: Schema meta is NULL");
                }

                std::string table = metadata.at(pos).info[4];

                const CassTableMeta *table_meta = cass_keyspace_meta_table_by_name(keyspace_meta, table.c_str());
                if (!table_meta || (cass_table_meta_column_count(table_meta) == 0)) {
                    throw ModuleException("Cache particles_table: constructor: Table meta is NULL");
                }

                std::vector<std::string> numpy_keys{"uuid"};
                std::vector<std::vector<std::string> > numpy_columns(2);

                numpy_columns[0] = {"data", metadata.at(pos).info[1], metadata.at(pos).info[2],
                                    metadata.at(pos).info[3]};
                numpy_columns[1] = {"position"};

                CacheTable *temp = NULL;


                uint64_t **uuid = (uint64_t **) values[0]->get_element(pos);


                cass_schema_meta_free(schema_meta);
                //TODO break down the token range
                try {
                    std::map<std::string, std::string> config;
                    temp = new CacheTable(table, std::string(keyspace), numpy_keys,
                                          numpy_columns, std::string("WHERE token(uuid)=>? AND token(uuid)<?"), {},
                                          session, config);
                } catch (ModuleException e) {
                    throw e;
                }

                void *payload = malloc(sizeof(uint64_t *));

                memcpy(payload, uuid, sizeof(uint64_t));

                TupleRow *uuid_key = new TupleRow(temp->keys_factory->get_metadata(), sizeof(uint64_t), payload);

                std::vector<const TupleRow *> npy_array = temp->get_crow(uuid_key);
                PyObject *py_list_array = temp->values_factory->tuples_as_py(npy_array);


                PyObject *py_array = PyList_GetItem(py_list_array, 0);



                /*** END MERGE ***/

                row = values_factory->tuples_as_py(values);
                PyList_SetItem(row, pos, py_array);



                /*** CLEANUP ***/
                delete (temp);
                //if the data is inserted inside the cache we cant call delete, it doesnt detect there is a copy inside the cache
                for (const TupleRow *block:values) {
                    delete (block);
                }
                for (const TupleRow *block:npy_array) {
                    delete (block);
                }
                delete (uuid_key);
                delete (keys);


            } else if (metadata.at(pos).info.size() == 4) {

                row = values_factory->tuples_as_py(values);

                //if the data is inserted inside the cache we cant call delete, it doesnt detect there is a copy inside the cache
                for (const TupleRow *block:values) {
                    delete (block);
                }
                delete (keys);
            } else {
                //skip
            }
        }
    } else {
        row = values_factory->tuples_as_py(values);

    }
    return row;
}


std::vector<const TupleRow *> CacheTable::get_crow(TupleRow *keys) {

    Poco::SharedPtr<TupleRow> ptrElem = myCache->get(*keys);
    if (!ptrElem.isNull()) {
        return std::vector<const TupleRow *>(1, ptrElem.get());
    }
    /* Not present on cache, a query is performed */
    CassStatement *statement = cass_prepared_bind(prepared_query);

    this->keys_factory->bind(statement, keys, 0);

    CassFuture *query_future = cass_session_execute(session, statement);
    const CassResult *result = cass_future_get_result(query_future);
    CassError rc = cass_future_error_code(query_future);
    if (result == NULL) {
        /* Handle error */
        printf("%s\n", cass_error_desc(rc));
        cass_future_free(query_future);
        cass_statement_free(statement);
        return std::vector<const TupleRow *>(0);
    }

    cass_future_free(query_future);
    cass_statement_free(statement);
    if (0 == cass_result_row_count(result)) {

        return std::vector<const TupleRow *>(0);
    }
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