#ifndef HFETCH_ARRAY_DATASTORE_H
#define HFETCH_ARRAY_DATASTORE_H


#include "SpaceFillingCurve.h"
#include "CacheTable.h"

#include <climits>
#include <list>
#include <set>

#include <arrow/memory_pool.h>
#include <arrow/io/file.h>
#include <arrow/status.h>
#include <arrow/ipc/reader.h>
#include <arrow/buffer.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/io/api.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/interfaces.h>
#include <arrow/ipc/feather.h>
#include "arrow/io/file.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"
#include "arrow/status.h"
#include "arrow/util/io_util.h"
#include <arrow/memory_pool.h>
#include <arrow/io/file.h>
#include <arrow/status.h>
#include <arrow/ipc/reader.h>
#include <arrow/buffer.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/builder_binary.h>

class ArrayDataStore {

public:

    ArrayDataStore(const char *table, const char *keyspace, CassSession *session,
                   std::map<std::string, std::string> &config);

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
    void read_numpy_from_cas_arrow(const uint64_t *storage_id, ArrayMetadata &metadata, std::list<std::vector<uint32_t> > &cols, void *save);

protected:

    void store_numpy_partition_into_cas(const uint64_t *storage_id , Partition part) const;
    void store_numpy_into_cas_as_arrow(const uint64_t *storage_id, ArrayMetadata &metadata,
                                       void *data) const;
    /* FIXME
	 * void store_numpy_into_cas_by_coords_as_arrow(const uint64_t *storage_id, ArrayMetadata &metadata,
                                                 void *data, std::list<std::vector<uint32_t> > &coord) const;
	*/


    CacheTable *cache = nullptr, *read_cache = nullptr;
    CacheTable *metadata_cache = nullptr, *metadata_read_cache=nullptr;
    CacheTable *cache_arrow = nullptr;

    SpaceFillingCurve partitioner;



};


#endif //HFETCH_ArrayDataStore_H
