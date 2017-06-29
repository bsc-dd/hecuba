#include "TableMetadata.h"

/***
 * Returns the allocation number of bytes required to allocate a type of data
 * @param VT Cassandra Type
 * @return Allocation size needed to store data of type VT
 */
uint16_t TableMetadata::compute_size_of(const CassValueType VT) const {
    switch (VT) {
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
            //TODO
            std::cerr << "Timestamp data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_UUID: {
            return sizeof(uint64_t*);
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
            std::cerr << "Date data type supported yet" << std::endl;

            //TODO
            break;
        }
        case CASS_VALUE_TYPE_TIME: {
            std::cerr << "Time data type supported yet" << std::endl;
            //TODO

            break;
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
            //TODO

            std::cerr << "Tuple data type supported yet" << std::endl;
            break;
        }
        case CASS_VALUE_TYPE_UDT:
        case CASS_VALUE_TYPE_CUSTOM:
        case CASS_VALUE_TYPE_UNKNOWN:
        default:
            throw ModuleException("Can't parse data: Unknown data type or user defined type");
            //TODO
    }
    return 0;
}


TableMetadata::TableMetadata(const char* table_name, const char* keyspace_name,
                             std::vector<std::map<std::string,std::string> > &keys_names,
                             std::vector<std::map<std::string,std::string> > &columns_names,
                             CassSession* session) {


    if (keys_names.empty()) throw ModuleException("TableMetadata: No keys received");

    uint32_t n_keys = (uint32_t) keys_names.size();
    uint32_t n_cols = (uint32_t) columns_names.size(); //TODO check for *
    for (uint32_t i= 0; i<n_keys; ++i) {
        std::transform(keys_names[i]["name"].begin(), keys_names[i]["name"].end(), keys_names[i]["name"].begin(), ::tolower);
    }
    for (uint32_t i= 0; i<n_cols; ++i) {
        std::transform(columns_names[i]["name"].begin(), columns_names[i]["name"].end(), columns_names[i]["name"].begin(), ::tolower);
    }
    this->table=std::string(table_name);
    std::transform(this->table.begin(), this->table.end(), this->table.begin(), ::tolower);
    this->keyspace=std::string(keyspace_name);
    std::transform(this->keyspace.begin(), this->keyspace.end(), this->keyspace.begin(), ::tolower);



    const CassSchemaMeta *schema_meta = cass_session_get_schema_meta(session);
    if (!schema_meta) {
        std::string error_msg = "TableMetadata constructor: Cassandra schema doesn't exist, probably not connected...";
        if (session==NULL) error_msg+= "session with cassandra not stablished";
        throw ModuleException(error_msg.c_str());
    }

    const CassKeyspaceMeta *keyspace_meta = cass_schema_meta_keyspace_by_name(schema_meta, this->keyspace.c_str());
    if (!keyspace_meta) {
        throw ModuleException("The keyspace "+std::string(keyspace_name)+" has no metadatas,"
                "check the keyspace name and make sure it exists");
    }


    const CassTableMeta *table_meta = cass_keyspace_meta_table_by_name(keyspace_meta, this->table.c_str());
    if (!table_meta || (cass_table_meta_column_count(table_meta)==0)) {
        throw ModuleException("The table "+std::string(table_name)+" has no metadatas,"
                " check the table name and make sure it exists");
    }

//TODO Switch to unordered maps for efficiency

    std::map<std::string, ColumnMeta> metadatas;

/*** build metadata ***/

    CassIterator *iterator = cass_iterator_columns_from_table_meta(table_meta);
    while (cass_iterator_next(iterator)) {
        const CassColumnMeta *cmeta = cass_iterator_get_column_meta(iterator);

        const char *value;
        size_t length;
        cass_column_meta_name(cmeta, &value, &length);

        const CassDataType *type = cass_column_meta_data_type(cmeta);

        metadatas[value]= {};
        metadatas[value].type = cass_data_type_type(type);
        metadatas[value].size = compute_size_of(metadatas[value].type);
        metadatas[value].col_type = cass_column_meta_type(cmeta);
    }
    cass_iterator_free(iterator);
    cass_schema_meta_free(schema_meta);



    std::string key = keys_names[0]["name"];
    if (key.empty()) throw ModuleException("Empty key name given on position 0");
    std::string select_where = key+"=? ";
    std::string keys = key;
    std::string col;
    std::string tokens_keys = "";
    if (metadatas[key].col_type==CASS_COLUMN_TYPE_PARTITION_KEY) tokens_keys+=key;

    for (uint16_t i = 1; i<n_keys; ++i){
        key = keys_names[i]["name"];
        if (key.empty()) throw ModuleException("Empty key name given on position: "+std::to_string(i));
        keys+=","+key;
        select_where+="AND "+key+"=? ";
        if (metadatas[key].col_type==CASS_COLUMN_TYPE_PARTITION_KEY) tokens_keys+=","+key;
    }
    if (tokens_keys.empty()) throw ModuleException("No partition key detected among the keys: "+keys);

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

    std::string select_tokens_where =" token("+tokens_keys+")>=? AND token("+tokens_keys+")<? ";

    select="SELECT "+cols+" FROM " + this->keyspace + "." + this->table + " WHERE "+select_where+";";
    select_keys_tokens="SELECT "+keys+" FROM " + this->keyspace + "." + this->table + " WHERE "+select_tokens_where+";";
    select_tokens_values="SELECT "+cols+" FROM " + this->keyspace + "." + this->table + " WHERE "+select_tokens_where+";";
    select_tokens_all="SELECT "+keys+","+cols+" FROM " + this->keyspace + "." + this->table + " WHERE "+select_tokens_where+";";

    insert="INSERT INTO "+ this->keyspace + "." + this->table + "("+keys+","+cols+")"+"VALUES (?";
    for (uint16_t i = 1; i<n_keys+n_cols;++i) {
        insert+=",?";
    }
    insert+=");";


    std::vector<ColumnMeta> keys_meta(n_keys);
    std::vector<ColumnMeta> cols_meta(n_cols);

    if (!columns_names.empty()) {
        cols_meta[0] = metadatas[columns_names[0]["name"]];
        cols_meta[0].info = columns_names[0];
        cols_meta[0].position = 0;
        for (uint16_t i = 1; i < cols_meta.size(); ++i) {
            cols_meta[i] = metadatas[columns_names[i]["name"]];
            cols_meta[i].info = columns_names[i];
            cols_meta[i].position = cols_meta[i - 1].position + cols_meta[i - 1].size;
        }
    }

    keys_meta[0]=metadatas[keys_names[0]["name"]];
    keys_meta[0].info=keys_names[0];
    keys_meta[0].position = 0;
    for (uint16_t i = 1; i < keys_meta.size(); ++i) {
        keys_meta[i]=metadatas[keys_names[i]["name"]];
        keys_meta[i].info=keys_names[i];
        keys_meta[i].position = keys_meta[i - 1].position + keys_meta[i - 1].size;
    }


    std::vector<ColumnMeta> items_meta(cols_meta);
    items_meta.insert(items_meta.begin(),keys_meta.begin(),keys_meta.end());
    uint16_t keys_offset = keys_meta[n_keys-1].position+ keys_meta[n_keys-1].size;

    for (uint16_t i = (uint16_t)keys_meta.size(); i< items_meta.size(); ++i){
        items_meta[i].position+=keys_offset;
    }

    this->cols=std::make_shared<std::vector<ColumnMeta> >(cols_meta);
    this->keys=std::make_shared<std::vector<ColumnMeta> >(keys_meta);
    this->items=std::make_shared<std::vector<ColumnMeta> >(items_meta);
}