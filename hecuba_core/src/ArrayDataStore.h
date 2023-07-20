#ifndef HFETCH_ARRAY_DATASTORE_H
#define HFETCH_ARRAY_DATASTORE_H


#include "SpaceFillingCurve.h"
#include "CacheTable.h"
#include "StorageInterface.h"
#include <set>

#include <string.h>


class ArrayDataStore {

public:

    ArrayDataStore(const char *table, const char *keyspace, std::shared_ptr<StorageInterface> storage,
                   std::map<std::string, std::string> &config);

    ArrayDataStore(const ArrayDataStore& src);
    ArrayDataStore& operator= (const ArrayDataStore& src);

    ~ArrayDataStore();

    void store_numpy_into_cas_by_coords(const uint64_t *storage_id, ArrayMetadata &metadata, void *data,
                                        std::list<std::vector<uint32_t> > &coord) const;

    void store_numpy_into_cas(const uint64_t *storage_id, ArrayMetadata &metadata, void *data) const;

    void read_numpy_from_cas_by_coords(const uint64_t *storage_id, ArrayMetadata &metadata,
                                       std::list<std::vector<uint32_t> > &coord, void *save);

    void read_numpy_from_cas(const uint64_t *storage_id, ArrayMetadata &metadata, void *save);

    // Returns the metadata of the array identified by the storage_id
    ArrayMetadata *read_metadata(const uint64_t *storage_id) const;

    // Overwrite the metadata of the array identified by the given storage_id
    //void update_metadata(const uint64_t *storage_id, ArrayMetadata *metadata) const;

    //lgarrobe
    std::string TN  = "";
    void read_numpy_from_cas_arrow(const uint64_t *storage_id, ArrayMetadata &metadata, std::vector<uint64_t> &cols, void *save);
    void store_numpy_into_cas_as_arrow(const uint64_t *storage_id, ArrayMetadata &metadata,
                                       void *data) const;
    void store_numpy_into_cas_by_cols_as_arrow(const uint64_t *storage_id, ArrayMetadata &metadata, void *data, std::vector<uint32_t> &cols) const;
    std::list<int32_t> get_cluster_ids(ArrayMetadata &metadata) const;
    std::list<std::tuple<uint64_t, uint32_t, uint32_t, std::vector<uint32_t>>> get_block_ids(ArrayMetadata &metadata) const;
    void wait_stores(void) const;

    CacheTable* getWriteCache(void) const ;
    CacheTable* getReadCache(void) const ;
    CacheTable* getMetaDataCache(void) const ;
protected:

    void store_numpy_partition_into_cas(const uint64_t *storage_id , Partition part) const;
    uint32_t get_row_elements(ArrayMetadata &metadata) const;


    static CacheTable* getStaticcache(const char *table_name, const char *keyspace_name,
                             std::vector<std::map<std::string, std::string>> &keys_names,
                             std::vector<std::map<std::string, std::string>> &columns_names,
                             CassSession* session,
                             std::map<std::string, std::string> &config); // This variable contains the cache table to be shared by all numpys if 'hecuba_sn_single_table' is set
    static CacheTable* getStaticHecubaIstorageCacheTable(const char *table_name, const char *keyspace_name,
                             std::vector<std::map<std::string, std::string>> &keys_names,
                             std::vector<std::map<std::string, std::string>> &columns_names,
                             CassSession *session,
                             std::map<std::string, std::string> &config);
    CacheTable *cache = nullptr, *read_cache = nullptr;
    CacheTable *metadata_cache = nullptr, *metadata_read_cache=nullptr;

    SpaceFillingCurve partitioner;

    bool has_lazy_writes = false; // flag to enable/disable lazy writes in cassandra (useful if there are multiple writes to the same object)
    bool arrow_enabled = false;
    bool arrow_optane  = false; // Intel OPTANE disk enabled?
    std::string arrow_path  = "";

    std::shared_ptr<StorageInterface> storage; //StorageInterface* storage;
private:
#ifdef ARROW
    int open_arrow_file(std::string arrow_file_name) ;
    int find_and_open_arrow_file(const uint64_t * storage_id, const uint32_t cluster_id, const std::string arrow_file_name);
    void *get_in_addr(struct sockaddr *sa);
#endif /*ARROW*/
    std::set<uint32_t> loaded_cluster_ids;
};


#endif //HFETCH_ArrayDataStore_H
