#include "TableMetadata.h"
#include <unistd.h>

/***
 * Returns the allocation number of bytes required to allocate a type of data
 * @param VT Cassandra Type
 * @return Allocation size needed to store data of type VT
 */
uint16_t TableMetadata::compute_size_of(const ColumnMeta &CM) const {
    switch (CM.type) {
        case CASS_VALUE_TYPE_TEXT:
        case CASS_VALUE_TYPE_VARCHAR:
        case CASS_VALUE_TYPE_ASCII: {
            return sizeof(char *);
        }
        case CASS_VALUE_TYPE_VARINT:
        case CASS_VALUE_TYPE_BIGINT: {
            return sizeof(int64_t);
        }
        case CASS_VALUE_TYPE_BLOB: {
            return sizeof(unsigned char *);
        }
        case CASS_VALUE_TYPE_BOOLEAN: {
            return sizeof(bool);
        }
        case CASS_VALUE_TYPE_COUNTER: {
            return sizeof(uint32_t);
        }
        case CASS_VALUE_TYPE_DECIMAL: {
            //TODO
            std::cerr << "Parse decimals data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_DOUBLE: {
            return sizeof(double);
        }
        case CASS_VALUE_TYPE_FLOAT: {
            return sizeof(float);
        }
        case CASS_VALUE_TYPE_INT: {
            return sizeof(int32_t);
        }
        case CASS_VALUE_TYPE_TIMESTAMP: {
            return sizeof(int64_t);
        }
        case CASS_VALUE_TYPE_UUID: {
            return sizeof(uint64_t *);
        }
        case CASS_VALUE_TYPE_TIMEUUID: {
            std::cerr << "TIMEUUID data type supported yet" << std::endl;
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_INET: {
            //TODO
            std::cerr << "INET data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_DATE: {
            return sizeof(int64_t);
        }
        case CASS_VALUE_TYPE_TIME: {
            return sizeof(int64_t);
        }
        case CASS_VALUE_TYPE_SMALL_INT: {
            return sizeof(int16_t);
        }
        case CASS_VALUE_TYPE_TINY_INT: {
            return sizeof(int8_t);
        }
        case CASS_VALUE_TYPE_LIST: {
            std::cerr << "List data type supported yet" << std::endl;
            //TODO
            break;
        }
        case CASS_VALUE_TYPE_MAP: {
            std::cerr << "Map data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_SET: {
            std::cerr << "Set data type supported yet" << std::endl;
            //TODO

            break;
        }
        case CASS_VALUE_TYPE_TUPLE: {
            return sizeof(void *);
        }
        case CASS_VALUE_TYPE_UDT: {
            //throw ModuleException("Can't parse data: User defined type not supported");
            return sizeof(ArrayMetadata *);
        }
        case CASS_VALUE_TYPE_CUSTOM:
            std::cerr << "Custom type" << std::endl;
            return sizeof(ArrayMetadata *);
        case CASS_VALUE_TYPE_UNKNOWN:
        default: {
            throw ModuleException("Can't parse data: Unknown data type or user defined type");
            //TODO
        }
    }
    return 0;
}


/***
 * Checks if all cassandra nodes agree on the schema. It may BLOCK the caller.
 * @return True if there is an agreement on the schema
 * Not in use right now
 */

#if 0
bool TableMetadata::checkSchemaAgreement(const CassSession *session) {

    CassSession * tmp = (CassSession *)session;
    static CassStatement * getLocalSchemaVersion = cass_statement_new("SELECT schema_version FROM system.local WHERE key='local'",0);
    static CassStatement * getPeersSchemaVersion = cass_statement_new("SELECT schema_version FROM system.peers",0);

    CassFuture *future_local = cass_session_execute(tmp, getLocalSchemaVersion);
    CassFuture *future_peers = cass_session_execute(tmp, getPeersSchemaVersion);

    CassError rc = cass_future_error_code(future_local);
    if (rc != CASS_OK) {
        /* Handle error */
        std::string error(cass_error_desc(rc));
        cass_future_free(future_local);
        throw ModuleException("TableMetadata: checkSchemaAgreement Get row error on result local" + error);
    }
    const CassResult *localresult = cass_future_get_result(future_local);

    rc = cass_future_error_code(future_peers);
    if (rc != CASS_OK) {
        /* Handle error */
        std::string error(cass_error_desc(rc));
        cass_future_free(future_peers);
        throw ModuleException("TableMetadata: checkSchemaAgreement Get row error on result peers" + error);
    }
    const CassResult *peer_result = cass_future_get_result(future_peers);


    CassUuid local_schema_version;
    const CassRow* rowLocal = cass_result_first_row(localresult);
    cass_value_get_uuid(cass_row_get_column_by_name(rowLocal, "schema_version"), &local_schema_version);
    cass_result_free(localresult);


    bool match = true;
    CassIterator *it = cass_iterator_from_result(peer_result);
    while (cass_iterator_next(it) && match) {
        CassUuid peer_schema_version;
        const CassRow *row = cass_iterator_get_row(it);
        cass_value_get_uuid(cass_row_get_column_by_name(row, "schema_version"), &peer_schema_version);
        match = ((local_schema_version.time_and_version   == peer_schema_version.time_and_version) &&
                (local_schema_version.clock_seq_and_node == peer_schema_version.clock_seq_and_node));
    }
    cass_iterator_free(it);
    cass_result_free(peer_result);

    return match;
}

#endif


std::map<std::string, ColumnMeta> TableMetadata::getMetaTypes(CassIterator *iterator) {
    std::map<std::string, ColumnMeta> metadatas;
    const char *value;
    size_t length;
    const CassDataType *type;
    while (cass_iterator_next(iterator)) {
        // Get column name and type
        const CassColumnMeta *cmeta = cass_iterator_get_column_meta(iterator);
        cass_column_meta_name(cmeta, &value, &length);
        type = cass_column_meta_data_type(cmeta);
        metadatas[value].type = cass_data_type_type(type);
        metadatas[value].col_type = cass_column_meta_type(cmeta);

        if (cass_data_type_type(type) == CASS_VALUE_TYPE_TUPLE) {
            uint32_t n_subtypes = (uint32_t) cass_data_type_sub_type_count(type);
            std::vector<ColumnMeta> v(n_subtypes);
            uint16_t offset = 0;
            // Get tuple elements information
            for (uint32_t subtype = 0; subtype < n_subtypes; ++subtype) {
                CassValueType cvt = cass_data_type_type(cass_data_type_sub_data_type(type, subtype));
                v[subtype].info["name"] = value;
                v[subtype].type = cvt;
                v[subtype].position = offset;
                v[subtype].size = compute_size_of(v[subtype]);
                offset += v[subtype].size;

            }
            metadatas[value].pointer = std::make_shared<std::vector<ColumnMeta>>(v);
        } else if (cass_data_type_type(type) == CASS_VALUE_TYPE_UDT) {
            const char *l_temp;
            size_t l_size;

            CassError rc = cass_data_type_type_name(type, &l_temp, &l_size);
            if (rc != CASS_OK) {
                std::cerr << cass_error_desc(rc) << std::endl;
                throw ModuleException("Can't fetch the user defined type from schema");
            }
            std::string type_name = std::string(l_temp, l_size);
            //if (type_name != "np_meta") throw ModuleException("Cassandra UDT not supported"); lgarrobe
            metadatas[value].dtype = type;
        }
    }
    return metadatas;
}

const CassTableMeta *TableMetadata::getCassTableMeta(const CassSession * session) {
    const CassSchemaMeta *schema_meta = cass_session_get_schema_meta(session);
    if (!schema_meta) {
        std::string error_msg = "TableMetadata constructor: Cassandra schema doesn't exist, probably not connected...";
        if (session == NULL) error_msg += "session with cassandra not stablished";
        throw ModuleException(error_msg.c_str());
    }

    const CassKeyspaceMeta *keyspace_meta = cass_schema_meta_keyspace_by_name(schema_meta, this->keyspace.c_str());
    if (!keyspace_meta) {
        throw ModuleException("The keyspace " + this->keyspace + " has no metadatas,"
                                                                             "check the keyspace name and make sure it exists");
    }


    const CassTableMeta *table_meta = cass_keyspace_meta_table_by_name(keyspace_meta, this->table.c_str());
    cass_schema_meta_free(schema_meta);

    return table_meta;
}

TableMetadata::TableMetadata(const char *table_name, const char *keyspace_name,
                             std::vector<std::map<std::string, std::string>> &keys_names,
                             std::vector<std::map<std::string, std::string>> &columns_names,
                             const CassSession *session) {


    if (keys_names.empty()) throw ModuleException("TableMetadata: No keys received");

    uint32_t n_keys = (uint32_t) keys_names.size();
    uint32_t n_cols = (uint32_t) columns_names.size(); //TODO check for *
    for (uint32_t i = 0; i < n_keys; ++i) {
        std::transform(keys_names[i]["name"].begin(), keys_names[i]["name"].end(), keys_names[i]["name"].begin(),
                       ::tolower);
    }
    for (uint32_t i = 0; i < n_cols; ++i) {
        std::transform(columns_names[i]["name"].begin(), columns_names[i]["name"].end(),
                       columns_names[i]["name"].begin(), ::tolower);
    }
    this->table = std::string(table_name);
    std::transform(this->table.begin(), this->table.end(), this->table.begin(), ::tolower);
    this->keyspace = std::string(keyspace_name);
    std::transform(this->keyspace.begin(), this->keyspace.end(), this->keyspace.begin(), ::tolower);


    const CassTableMeta *table_meta = getCassTableMeta(session);
    if (!table_meta || (cass_table_meta_column_count(table_meta) == 0)) {
        uint32_t i = 0;

        do {// wait until all nodes agree with the schema
            sleep(1);
            table_meta=getCassTableMeta(session);
            i++;
        }
        while ( (i<20) &&  (!table_meta || (cass_table_meta_column_count(table_meta) == 0)));
        if (i==20) {
            throw ModuleException("The table " + std::string(table_name) + " has no metadatas,"
                    " check the table name and make sure it exists");
        }
    }

//TODO Switch to unordered maps for efficiency

    CassIterator *iterator = cass_iterator_columns_from_table_meta(table_meta);

    std::map<std::string, ColumnMeta> metadatas = getMetaTypes(iterator);
    cass_iterator_free(iterator);
    //cass_schema_meta_free(schema_meta);

    std::string key = keys_names[0]["name"];
    if (key.empty()) throw ModuleException("Empty key name given on position 0");
    std::string select_where = key + "=? ";
    std::string keys = key;
    std::string col;
    std::string tokens_keys = "";
    if (metadatas[key].col_type == CASS_COLUMN_TYPE_PARTITION_KEY) tokens_keys += key;

    for (uint16_t i = 1; i < n_keys; ++i) {
        key = keys_names[i]["name"];
        if (key.empty()) throw ModuleException("Empty key name given on position: " + std::to_string(i));
        keys += "," + key;
        select_where += "AND " + key + "=? ";
        if (metadatas[key].col_type == CASS_COLUMN_TYPE_PARTITION_KEY) {
            if (!tokens_keys.empty()) tokens_keys += ",";
            tokens_keys += key;
        }
    }
    if (tokens_keys.empty()) throw ModuleException("No partition key detected among the keys: " + keys);

    std::string cols = "";
    if (!columns_names.empty()) {
        cols += columns_names[0]["name"]; //TODO Check for *
        if (cols.empty()) throw ModuleException("Empty column name given on position 0");
        for (uint16_t i = 1; i < n_cols; ++i) {
            col = columns_names[i]["name"];
            if (col.empty()) throw ModuleException("Empty column name given onposition: " + std::to_string(i));
            cols += "," + col;
        }
    }
    std::string keys_and_cols = keys;
    if (!cols.empty()) keys_and_cols += ", " + cols;
    else cols = keys;

    std::string select_tokens_where = " token(" + tokens_keys + ")>=? AND token(" + tokens_keys + ")<? ";
    select = "SELECT " + cols + " FROM " + this->keyspace + "." + this->table + " WHERE " + select_where + ";";

    select_keys_tokens =
            "SELECT " + keys + " FROM " + this->keyspace + "." + this->table + " WHERE " + select_tokens_where + ";";
    select_tokens_values =
            "SELECT " + cols + " FROM " + this->keyspace + "." + this->table + " WHERE " + select_tokens_where + ";";

    select_tokens_all = "SELECT " + keys_and_cols + " FROM " + this->keyspace + "." + this->table + " WHERE " +
                        select_tokens_where + ";";

    partial_insert = "INSERT INTO " + this->keyspace + "." + this->table + "(" + keys;

    insert = "INSERT INTO " + this->keyspace + "." + this->table + "(" + keys_and_cols + ")" + "VALUES (?";
    for (uint16_t i = 1; i < n_keys + n_cols; ++i) {
        insert += ",?";
    }
    insert += ");";

    delete_row = "DELETE  FROM " + this->keyspace + "." + this->table + " WHERE " + select_where + ";";

    std::vector<ColumnMeta> keys_meta(n_keys);
    std::vector<ColumnMeta> cols_meta(n_cols);

    if (!columns_names.empty()) {
        cols_meta[0] = metadatas[columns_names[0]["name"]];
        cols_meta[0].info = columns_names[0];
        cols_meta[0].size = compute_size_of(cols_meta[0]);
        cols_meta[0].position = 0;
        for (uint16_t i = 1; i < cols_meta.size(); ++i) {
            cols_meta[i] = metadatas[columns_names[i]["name"]];
            cols_meta[i].info = columns_names[i];
            cols_meta[i].size = compute_size_of(cols_meta[i]);
            cols_meta[i].position = cols_meta[i - 1].position + cols_meta[i - 1].size;
        }
    }

    keys_meta[0] = metadatas[keys_names[0]["name"]];
    keys_meta[0].info = keys_names[0];
    keys_meta[0].size = compute_size_of(keys_meta[0]);
    keys_meta[0].position = 0;
    for (uint16_t i = 1; i < keys_meta.size(); ++i) {
        keys_meta[i] = metadatas[keys_names[i]["name"]];
        keys_meta[i].info = keys_names[i];
        keys_meta[i].size = compute_size_of(keys_meta[i]);
        keys_meta[i].position = keys_meta[i - 1].position + keys_meta[i - 1].size;
    }


    std::vector<ColumnMeta> items_meta(cols_meta);
    items_meta.insert(items_meta.begin(), keys_meta.begin(), keys_meta.end());
    uint16_t keys_offset = keys_meta[n_keys - 1].position + keys_meta[n_keys - 1].size;

    for (uint16_t i = (uint16_t) keys_meta.size(); i < items_meta.size(); ++i) {
        items_meta[i].position += keys_offset;
    }

    this->cols = std::make_shared<std::vector<ColumnMeta> >(cols_meta);
    this->keys = std::make_shared<std::vector<ColumnMeta> >(keys_meta);
    this->items = std::make_shared<std::vector<ColumnMeta> >(items_meta);
}

std::shared_ptr<const std::vector<ColumnMeta> > TableMetadata::get_single_value(const char *value_name) const {
    // TODO : Add a hash map to cache 'value_name' ColumnMeta and avoid searching
    std::string value(value_name);

    std::vector<ColumnMeta> res(1);

    ColumnMeta m;
    for (uint16_t i = 0; i < cols->size(); ++i) {
        m = (*cols)[i];
        if (m.info["name"] == value) {
            res[0] =  m;
            res[0].position = 0; // Ignore other elements in the row, now it will ALWAYS be at the first position
            return std::make_shared<std::vector<ColumnMeta>>(res);
        }
    }
    throw ModuleException("get_single_value: Unknown column name [" + value + "]");
}

// completes the build of the insert query for just one attribute
const char *TableMetadata::get_partial_insert_query(const std::string &attr_name) const {
    uint32_t n_keys = (uint32_t) keys->size();
    std::string insert = partial_insert
                            + "," + attr_name + ")" + "VALUES (?";
    for (uint16_t i = 1; i < n_keys + 1; ++i) {
        insert += ",?";
    }
    insert += ");";

    char * mistring = (char*) malloc (insert.size()+1);
    strncpy(mistring, insert.c_str(), insert.size()+1);
    mistring[insert.size()] = '\0';
    return mistring;
}

/** Returns a pair with partition_keys size, clustering_keys size */
std::pair<uint16_t, uint16_t> TableMetadata::get_keys_size(void) const {
    ColumnMeta key;
    int partKeySize = 0;
    int clustKeySize = 0;

    for (uint16_t i = 0; i < keys->size(); ++i) {
        key = (*keys)[i];
        //for(auto k:key.info) {
        //    std::cout<< "DEBUG: TableMetadata::get_keys_size "<< k.first << " " << k.second<<std::endl;
        //}
        //std::cout<< "DEBUG: TableMetadata::get_keys_size col_type "<< key.col_type <<std::endl;
        //std::cout<< "DEBUG: TableMetadata::get_keys_size position "<< key.position <<std::endl;
        //std::cout<< "DEBUG: TableMetadata::get_keys_size size "<< key.size <<std::endl;

        if (key.col_type == CASS_COLUMN_TYPE_PARTITION_KEY) {
            partKeySize += key.size;
        } else if (key.col_type == CASS_COLUMN_TYPE_CLUSTERING_KEY) {
            clustKeySize += key.size;
        }
    }
    return std::pair<uint16_t, uint16_t>(partKeySize, clustKeySize);
}

/** Returns the values's size */
uint32_t TableMetadata::get_values_size(void) const {
    ColumnMeta value;
    uint32_t size = 0;

    for (uint16_t i = 0; i < cols->size(); ++i) {
        value = (*cols)[i];

        size += value.size;
    }
    return size;
}

/** Return the size of column 'pos' element */
uint32_t TableMetadata::get_values_size(int pos) const {
    return (*cols)[pos].size;
}
