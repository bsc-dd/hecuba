#ifndef HFETCH_TABLEMETADATA_H
#define HFETCH_TABLEMETADATA_H

#include <cassandra.h>
#include <cstdint>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>
#include <map>


#include "ModuleException.h"
#include "SpaceFillingCurve.h"

struct ColumnMeta {
    ColumnMeta() = default;

    ColumnMeta(const ColumnMeta &CM) {
            *this = CM;
    }

    ColumnMeta& operator=(const ColumnMeta& CM) {
        this->info = CM.info;
        this->type = CM.type;
        this->dtype = CM.dtype;
        this->position = CM.position;
        this->size = CM.size;
        this->col_type = CM.col_type;
        this->pointer = CM.pointer;
        return *this;
    }

    ColumnMeta(std::map<std::string, std::string> &info, CassValueType cv_type, CassDataType *cd_type, uint16_t offset, uint16_t bsize) {
        this->info = info;
        this->type = cv_type;
        this->dtype = cd_type;
        this->position = offset;
        this->size = bsize;
        this->col_type = CASS_COLUMN_TYPE_REGULAR;
    }

    uint16_t position, size;
    CassValueType type;
    const CassDataType *dtype;
    CassColumnType col_type;
    std::map<std::string, std::string> info;
    std::shared_ptr<std::vector<ColumnMeta> > pointer;


};


class TableMetadata {


public:

    TableMetadata(const char *table_name, const char *keyspace_name,
                  std::vector<std::map<std::string, std::string>> &keys_names,
                  std::vector<std::map<std::string, std::string>> &columns_names,
                  const CassSession *session);


    std::shared_ptr<const std::vector<ColumnMeta> > get_keys() const {
        return keys;
    }

    std::shared_ptr<const std::vector<ColumnMeta> > get_values() const {
        return cols;
    }

    std::shared_ptr<const std::vector<ColumnMeta> > get_single_value(const char *value_name) const;

    const ColumnMeta *get_single_column(const std::string &value_name) const;
    const ColumnMeta *get_single_key(const std::string &key_name) const;

    std::shared_ptr<const std::vector<ColumnMeta> > get_items() const {
        return items;
    }

    const char *get_select_query() const {
        return select.c_str();
    }

    const char *get_select_all_tokens() const {
        return select_tokens_all.c_str();
    }

    const char *get_select_values_tokens() const {
        return select_tokens_values.c_str();
    }

    const char *get_select_keys_tokens() const {
        return select_keys_tokens.c_str();
    };

    // completes the build of the insert query for just one attribute
    const char *get_partial_insert_query( const std::string &attr_name) const ;

    const char *get_insert_query() const {
        return insert.c_str();
    }

    const char *get_delete_query() const {
        return delete_row.c_str();
    }

    const char *get_table_name() const {
        return table.c_str();
    }

    const char *get_keyspace() const {
        return this->keyspace.c_str();
    }


    std::pair<uint16_t, uint16_t> get_keys_size(void) const;
    uint32_t get_values_size(void) const;
    uint32_t get_values_size(int pos) const;
    uint32_t get_columnname_position(const std::string &columnname) const;
    uint32_t get_keyname_position(const std::string &columnname) const;

private:
    uint16_t compute_size_of(const ColumnMeta &CM) const;

    std::map<std::string, ColumnMeta> getMetaTypes(CassIterator *iterator);

    //uint32_t total_bytes;
    std::shared_ptr<const std::vector<ColumnMeta> > cols;
    std::shared_ptr<const std::vector<ColumnMeta> > keys;
    std::shared_ptr<const std::vector<ColumnMeta> > items;
    std::string keyspace, table;
    std::string select, insert, select_tokens_all, select_tokens_values, select_keys_tokens, delete_row;
    std::string partial_insert;
    //bool checkSchemaAgreement(const CassSession *session);
    const CassTableMeta *getCassTableMeta(const CassSession * session);

};


#endif //HFETCH_TABLEMETADATA_H
