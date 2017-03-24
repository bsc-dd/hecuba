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
            //TODO

            std::cerr << "UUID data type supported yet" << std::endl;
            break;
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
            throw ModuleException("Unknown data type, can't parse");
            //TODO
    }
    return 0;
}


TableMetadata::TableMetadata(const char* table_name, const char* keyspace_name,
                             std::vector <std::string>& keys_names,std::vector <std::string>& columns_names,
                             CassSession* session) {


    if (columns_names.empty()) throw ModuleException("TableMetadata: 0 columns names received");

    this->table=std::string(table_name);
    this->keyspace=std::string(keyspace_name);


    select_tokens="SELECT "+keys_names[0];
    for (uint16_t i = 1; i<keys_names.size(); ++i){
        select_tokens+=","+keys_names[i];
    }

    insert="INSERT INTO "+ this->keyspace + "." + this->table + "("+columns_names[0];
    select="SELECT "+columns_names[0];
    select_tokens+=","+columns_names[0]; //TODO case where columns has 0 elements
    for (uint16_t i = 1; i<columns_names.size(); ++i){
        select+=","+columns_names[i];
        select_tokens+=","+columns_names[i];
        insert+=","+columns_names[i];
    }
    select+= " FROM "+ this->keyspace + "." + this->table + " ";
    select_tokens+= " FROM "+ this->keyspace + "." + this->table + " ";
    select+=" WHERE "+keys_names[0]+"=? ";

    for (uint16_t i = 1; i<keys_names.size(); ++i){
        select+="AND "+keys_names[i]+"=? ";
        insert+=","+keys_names[i];
    }
    insert+=") VALUES (?";
    for (uint16_t i = 0; i<keys_names.size()+columns_names.size();++i) {
        insert+=",?";
    }
    select_tokens+=" WHERE token("+keys_names[0]+")>=? AND token("+keys_names[0]+")<?;"; //TODO allow complex partition key
    insert+=");";
    select+=";";


    const CassSchemaMeta *schema_meta = cass_session_get_schema_meta(session);
    if (!schema_meta) {
        throw ModuleException("Cache particles_table: constructor: Schema meta is NULL");
    }

    const CassKeyspaceMeta *keyspace_meta = cass_schema_meta_keyspace_by_name(schema_meta, keyspace_name);
    if (!keyspace_meta) {
        throw ModuleException("Keyspace particles_table: constructor: Schema meta is NULL");
    }


    const CassTableMeta *table_meta = cass_keyspace_meta_table_by_name(keyspace_meta, table_name);
    if (!table_meta || (cass_table_meta_column_count(table_meta)==0)) {
        throw ModuleException("Cache particles_table: constructor: Table meta is NULL");
    }


    uint32_t n_cols = (uint32_t) columns_names.size();
    uint32_t n_keys = (uint32_t) keys_names.size();

    if (columns_names[0] == "*") n_cols = (uint32_t) cass_table_meta_column_count(table_meta) - n_keys;

    std::vector<ColumnMeta> cols_meta = std::vector< ColumnMeta>(n_cols);
    std::vector<ColumnMeta> keys_meta = std::vector< ColumnMeta>(n_keys);

/*** build metadata ***/
// TODO its more efficient having all the columns data stored and then make subsets

    CassIterator *iterator = cass_iterator_columns_from_table_meta(table_meta);
    while (cass_iterator_next(iterator)) {
        const CassColumnMeta *cmeta = cass_iterator_get_column_meta(iterator);

        const char *value;
        size_t length;
        cass_column_meta_name(cmeta, &value, &length);

        const CassDataType *type = cass_column_meta_data_type(cmeta);
        std::string meta_col_name(value);
//if meta col name is inside cols names, keys set

        for (uint16_t j=0; j<columns_names.size(); ++j) {
            const std::string ss = columns_names[j];
            if (meta_col_name == ss || columns_names[0] == "*") {
                cols_meta[j] ={value,cass_data_type_type(type)};
                cols_meta[j].size = compute_size_of( cols_meta[j].type);
                break;
            }
        }
        for (uint16_t j=0; j<keys_names.size(); ++j) {
            const std::string ss = keys_names[j];
            if (meta_col_name == ss || keys_names[0] == "*") {
                keys_meta[j] ={value,cass_data_type_type(type)};
                keys_meta[j].size = compute_size_of(keys_meta[j].type);
                break;
            }
        }


    }
    cass_iterator_free(iterator);
    cass_schema_meta_free(schema_meta);

    cols_meta[0].position=0;
    for (uint16_t i = 1; i<cols_meta.size(); ++i) {
        cols_meta[i].position=cols_meta[i-1].position+cols_meta[i-1].size;
    }
    keys_meta[0].position=0;
    for (uint16_t i = 1; i<keys_meta.size(); ++i) {
        keys_meta[i].position=keys_meta[i-1].position + keys_meta[i-1].size;
    }

    std::vector<ColumnMeta> items_meta(cols_meta);
    items_meta.insert(items_meta.begin(),keys_meta.begin(),keys_meta.end());
    uint16_t keys_offset = keys_meta[keys_meta.size()-1].position+ keys_meta[keys_meta.size()-1].size;

    for (uint16_t i = (uint16_t)keys_meta.size(); i< items_meta.size(); ++i){
        items_meta[i].position+=keys_offset;
    }

    this->cols=std::make_shared<std::vector<ColumnMeta> >(cols_meta);
    this->keys=std::make_shared<std::vector<ColumnMeta> >(keys_meta);
    this->items=std::make_shared<std::vector<ColumnMeta> >(items_meta);
}