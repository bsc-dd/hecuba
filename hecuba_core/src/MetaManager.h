#ifndef HECUBA_CORE_METAMANAGER_H
#define HECUBA_CORE_METAMANAGER_H

#include "string"
#include "Writer.h"
#include "SpaceFillingCurve.h"
#include "ModuleException.h"

class MetaManager {

public:

    MetaManager(const TableMetadata *table_meta, CassSession *session,
                std::map<std::string, std::string> &config);

    ~MetaManager();

    void create_data_model(const std::string &create_table_query) const;


    void register_obj(const uint64_t *storage_id, const std::string &name, const ArrayMetadata &numpy_meta) const;

private:
    Writer *writer;
};


#endif //HECUBA_CORE_METAMANAGER_H
