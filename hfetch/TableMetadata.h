#ifndef HFETCH_TABLEMETADATA_H
#define HFETCH_TABLEMETADATA_H

#include <cassandra.h>
#include <cstdint>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>
#include <map>


#include "ModuleException.h"


struct ColumnMeta {
    ColumnMeta() {}

    ColumnMeta(std::map<std::string, std::string> &info, CassValueType cv_type, uint16_t offset, uint16_t bsize) {
        this->info = info;
        this->type = cv_type;
        this->position = offset;
        this->size = bsize;
        col_type = CASS_COLUMN_TYPE_REGULAR;
    }

    uint16_t position, size;
    CassValueType type;
    CassColumnType col_type;
    std::map<std::string, std::string> info;
};


class TableMetadata {


public:

    TableMetadata(const char *table_name, const char *keyspace_name,
                  std::vector<std::map<std::string, std::string>> &keys_names,
                  std::vector<std::map<std::string, std::string>> &columns_names,
                  CassSession *session);

    std::shared_ptr<const std::vector<ColumnMeta> > get_keys() const {
        return keys;
    }

    std::shared_ptr<const std::vector<ColumnMeta> > get_values() const {
        return cols;
    }

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

private:
    uint16_t compute_size_of(const CassValueType VT) const;

    //uint32_t total_bytes;
    std::shared_ptr<const std::vector<ColumnMeta> > cols;
    std::shared_ptr<const std::vector<ColumnMeta> > keys;
    std::shared_ptr<const std::vector<ColumnMeta> > items;
    std::string keyspace, table;
    std::string select, insert, select_tokens_all, select_tokens_values, select_keys_tokens, delete_row;

};


#endif //HFETCH_TABLEMETADATA_H
