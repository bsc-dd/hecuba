#include "ArrayDataStore.h"


ArrayDataStore::ArrayDataStore(CacheTable *cache, CacheTable *read_cache,
                               std::map<std::string, std::string> &config) {

    //this->storage = storage;
/*

    std::vector <config_map> keysnames = {{{"name", "storage_id"}},
                                          {{"name", "cluster_id"}},
                                          {{"name", "block_id"}}};


    std::vector <config_map> colsnames = {{{"name", "payload"}}};
*/

    this->cache = cache;
    this->read_cache = read_cache;
    //this->storage->make_cache(table_meta, config);
}


ArrayDataStore::~ArrayDataStore() {

};



/***
 * Stores the array metadata by setting the cluster and block ids to -1. Deletes the array metadata afterwards.
 * @param storage_id UUID used as part of the key
 * @param np_metas ArrayMetadata
 */

void ArrayDataStore::update_metadata(const uint64_t *storage_id, ArrayMetadata *metadata) const {
    uint32_t offset = 0, keys_size = sizeof(uint64_t *) + sizeof(int32_t) * 2;

    int32_t cluster_id = -1, block_id = -1;

    char *keys = (char *) malloc(keys_size);
    //UUID
    uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);//new uint64_t[2];
    c_uuid[0] = *storage_id;
    c_uuid[1] = *(storage_id + 1);
    // [0] = storage_id.time_and_version;
    // [1] = storage_id.clock_seq_and_node;
    memcpy(keys, &c_uuid, sizeof(uint64_t *));
    offset = sizeof(uint64_t *);
    //Cluster id
    memcpy(keys + offset, &cluster_id, sizeof(int32_t));
    offset += sizeof(int32_t);
    //Block id
    memcpy(keys + offset, &block_id, sizeof(int32_t));




    //COPY VALUES
    offset = 0;
    void *values = (char *) malloc(sizeof(char *));

    //size of the vector of dims
    uint64_t size = sizeof(uint32_t) * metadata->dims.size();

    //plus the other metas
    size += sizeof(metadata->elem_size) + sizeof(metadata->inner_type) + sizeof(metadata->partition_type);

    //allocate plus the bytes counter
    unsigned char *byte_array = (unsigned char *) malloc(size + sizeof(uint64_t));

    // Copy num bytes
    memcpy(byte_array, &size, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    //copy everything from the metas
    memcpy(byte_array + offset, &metadata->elem_size, sizeof(metadata->elem_size));
    offset += sizeof(metadata->elem_size);
    memcpy(byte_array + offset, &metadata->inner_type, sizeof(metadata->inner_type));
    offset += sizeof(metadata->inner_type);
    memcpy(byte_array + offset, &metadata->partition_type, sizeof(metadata->partition_type));
    offset += sizeof(metadata->partition_type);
    memcpy(byte_array + offset, metadata->dims.data(), sizeof(uint32_t) * metadata->dims.size());


    // Copy ptr to bytearray
    memcpy(values, &byte_array, sizeof(unsigned char *));

    // Finally, we write the data
    cache->put_crow(keys, values);
}

/***
 * Write a complete numpy ndarray by using the partitioning mechanism defined in the metadata
 * @param storage_id identifying the numpy ndarray
 * @param np_metas ndarray characteristics
 * @param numpy to be saved into storage
 */
void ArrayDataStore::store(const uint64_t *storage_id, ArrayMetadata* metadata, void *data) const {

    SpaceFillingCurve::PartitionGenerator *partitions_it = this->partitioner.make_partitions_generator(metadata, data);

    char *keys = nullptr;
    void *values = nullptr;
    uint32_t offset = 0, keys_size = sizeof(uint64_t *) + sizeof(int32_t) * 2;
    uint64_t *c_uuid = nullptr;
    uint32_t half_int = 0;//(uint32_t)-1 >> (sizeof(uint32_t)*CHAR_BIT/2); //TODO be done properly
    int32_t cluster_id, block_id;
    while (!partitions_it->isDone()) {
        Partition part = partitions_it->getNextPartition();
        keys = (char *) malloc(keys_size);
        //UUID
        c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);//new uint64_t[2];
        c_uuid[0] = *storage_id;
        c_uuid[1] = *(storage_id + 1);
        // [0] = storage_id.time_and_version;
        // [1] = storage_id.clock_seq_and_node;
        memcpy(keys, &c_uuid, sizeof(uint64_t *));
        offset = sizeof(uint64_t *);
        //Cluster id
        cluster_id = part.cluster_id - half_int;
        memcpy(keys + offset, &cluster_id, sizeof(int32_t));
        offset += sizeof(int32_t);
        //Block id
        block_id = part.block_id - half_int;
        memcpy(keys + offset, &block_id, sizeof(int32_t));
        //COPY VALUES

        values = (char *) malloc(sizeof(char *));
        memcpy(values, &part.data, sizeof(char *));
        //FINALLY WE WRITE THE DATA
        cache->put_crow(keys, values);
    }
    //this->partitioner.serialize_metas();
    delete (partitions_it);

    // No need to flush the elements because the metadata are written after the data thanks to the queue
}


/***
 * Reads the metadata from the storage as an ArrayMetadata for latter use
 * @param storage_id identifing the numpy ndarray
 * @param cache
 * @return
 */
ArrayMetadata *ArrayDataStore::read_metadata(const uint64_t *storage_id) const {
    // Get metas from Cassandra
    int32_t cluster_id = -1, block_id = -1;

    char *buffer = (char *) malloc(sizeof(uint64_t*)+sizeof(int32_t)*2);
    // UUID
    uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);
    c_uuid[0] = *storage_id;
    c_uuid[1] = *(storage_id + 1);
    // Copy uuid
    memcpy(buffer, &c_uuid, sizeof(uint64_t *));
    int32_t offset = sizeof(uint64_t *);
    // Cluster id
    memcpy(buffer + offset, &cluster_id, sizeof(int32_t));
    offset += sizeof(int32_t);
    // Cluster id
    memcpy(buffer + offset, &block_id, sizeof(int32_t));


    // We fetch the data
    std::vector<const TupleRow *> results = cache->get_crow(buffer);

    if (results.empty()) throw ModuleException("Metadata for the array can't be found");

    // pos 0 is block id, pos 1 is payload
    const unsigned char *payload = *((const unsigned char **) (results[0]->get_element(0)));

    const uint64_t num_bytes = *((uint64_t *) payload);
    // Move pointer to the beginning of the data
    payload = payload + sizeof(num_bytes);


    uint32_t bytes_offset = 0;
    ArrayMetadata *arr_metas = new ArrayMetadata();
    // Load data
    memcpy(&arr_metas->elem_size, payload, sizeof(arr_metas->elem_size));
    bytes_offset += sizeof(arr_metas->elem_size);
    memcpy(&arr_metas->inner_type, payload + bytes_offset, sizeof(arr_metas->inner_type));
    bytes_offset += sizeof(arr_metas->inner_type);
    memcpy(&arr_metas->partition_type, payload + bytes_offset, sizeof(arr_metas->partition_type));
    bytes_offset += sizeof(arr_metas->partition_type);

    uint64_t nbytes = num_bytes - bytes_offset;
    uint32_t nelem = (uint32_t) nbytes / sizeof(uint32_t);
    if (nbytes % sizeof(uint32_t) != 0) throw ModuleException("something went wrong reading the dims of a numpy");
    arr_metas->dims = std::vector<uint32_t>(nelem);
    memcpy(arr_metas->dims.data(), payload + bytes_offset, nbytes);

    for (const TupleRow* &v : results) delete(v);
    return arr_metas;
}

void *ArrayDataStore::read_n_coord(const uint64_t *storage_id, ArrayMetadata *metadata, std::vector<std::pair<int,int> > coord) const {
    std::shared_ptr<const std::vector<ColumnMeta> > keys_metas = read_cache->get_metadata()->get_keys();
    uint32_t keys_size = (*--keys_metas->end()).size + (*--keys_metas->end()).position;

    std::vector<const TupleRow *> result, all_results;
    std::vector<Partition> all_partitions;

    uint64_t *c_uuid = nullptr;
    char *buffer = nullptr;
    int32_t cluster_id = 0, cluster_id1 = 0, cluster_id2 = 0, offset = 0;
    int32_t *block = nullptr;
    int32_t half_int = 0;//-1 >> sizeof(int32_t)/2; //TODO be done properly

    SpaceFillingCurve::PartitionGenerator *partitions_it = this->partitioner.make_partitions_generator(metadata,
                                                                                                         nullptr);
    //Convert
    int count;
    for(int i = 0; i < coord.size(); ++i) {
        cluster_id1 = std::ceil(coord[i].first / (BLOCK_SIZE * 4));
        cluster_id2 = std::ceil(coord[i].second / (BLOCK_SIZE * 4));
        count = 0;
        while (count < cluster_id1) {
            partitions_it->computeNextClusterId();
            ++count;
        }
        count = 0;
        while (count != cluster_id2) {
            cluster_id = partitions_it->computeNextClusterId();
            buffer = (char *) malloc(keys_size);
            //UUID
            c_uuid = new uint64_t[2]{*storage_id, *(storage_id + 1)};
            //[0] time_and_version;
            //[1] clock_seq_and_node;
            memcpy(buffer, &c_uuid, sizeof(uint64_t *));
            offset = sizeof(uint64_t *);
            //Cluster id
            memcpy(buffer + offset, &cluster_id, sizeof(cluster_id));
            //We fetch the data
            result = read_cache->get_crow(new TupleRow(keys_metas, keys_size, buffer));
            //build cluster
            all_results.insert(all_results.end(), result.begin(), result.end());
            for (const TupleRow *row:result) {
                block = (int32_t *) row->get_element(0);
                char **chunk = (char **) row->get_element(1);
                all_partitions.emplace_back(
                        Partition((uint32_t) cluster_id + half_int, (uint32_t) *block + half_int, *chunk));
            }
            ++count;
        }
    }

    if (all_partitions.empty()) {
        throw ModuleException("no npy found on sys");
    }

    void *data = partitions_it->merge_partitions(metadata, all_partitions);

    for (const TupleRow *item:all_results) delete (item);

    delete(partitions_it);

    return data;

}

/***
 * Reads a numpy ndarray by fetching the clusters indipendently
 * @param storage_id of the array to retrieve
 * @return Numpy ndarray as a Python object
 */
void *ArrayDataStore::read(const uint64_t *storage_id, ArrayMetadata *metadata) const {

    std::shared_ptr<const std::vector<ColumnMeta> > keys_metas = read_cache->get_metadata()->get_keys();
    uint32_t keys_size = (*--keys_metas->end()).size + (*--keys_metas->end()).position;



    std::vector<const TupleRow *> result, all_results;
    std::vector<Partition> all_partitions;

    uint64_t *c_uuid = nullptr;
    char *buffer = nullptr;
    int32_t cluster_id = 0, offset = 0;
    int32_t *block = nullptr;
    int32_t half_int = 0;//-1 >> sizeof(int32_t)/2; //TODO be done properly

    SpaceFillingCurve::PartitionGenerator *partitions_it = this->partitioner.make_partitions_generator(metadata,
                                                                                                       nullptr);

    while (!partitions_it->isDone()) {
        cluster_id = partitions_it->computeNextClusterId();
        buffer = (char *) malloc(keys_size);
        //UUID
        c_uuid = new uint64_t[2]{*storage_id, *(storage_id + 1)};
        //[0] time_and_version;
        //[1] clock_seq_and_node;
        memcpy(buffer, &c_uuid, sizeof(uint64_t *));
        offset = sizeof(uint64_t *);
        //Cluster id
        memcpy(buffer + offset, &cluster_id, sizeof(cluster_id));
        //We fetch the data
        result = read_cache->get_crow(new TupleRow(keys_metas, keys_size, buffer));
        //build cluster
        all_results.insert(all_results.end(), result.begin(), result.end());
        for (const TupleRow *row:result) {
            block = (int32_t *) row->get_element(0);
            char **chunk = (char **) row->get_element(1);
            all_partitions.emplace_back(
                    Partition((uint32_t) cluster_id + half_int, (uint32_t) *block + half_int, *chunk));
        }
    }



    if (all_partitions.empty()) {
        throw ModuleException("no npy found on sys");
    }

    void *data = partitions_it->merge_partitions(metadata, all_partitions);

    for (const TupleRow *item:all_results) delete (item);

    delete(partitions_it);

    return data;
}
