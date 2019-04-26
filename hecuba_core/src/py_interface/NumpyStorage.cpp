#include "NumpyStorage.h"


NumpyStorage::NumpyStorage(const TableMetadata *table_meta, std::shared_ptr<StorageInterface> storage,
                           std::map<std::string, std::string> &config) {
    this->storage = storage;
    this->writer = this->storage->make_writer(table_meta, config);
}


NumpyStorage::~NumpyStorage() {
    if (writer) {
        delete (writer->get_metadata());
        delete (writer);
    }
};


void NumpyStorage::store(const uint64_t *storage_id, PyArrayObject *numpy) const {
    ArrayMetadata *np_metas = get_np_metadata(numpy);
    np_metas->partition_type = ZORDER_ALGORITHM;
    store_entire_array(storage_id, np_metas, numpy);
    store_array_meta(storage_id, np_metas);
    delete (np_metas);
}


/***
 * Stores the array metadata by setting the cluster and block ids to -1. Deletes the array metadata afterwards.
 * @param storage_id UUID used as part of the key
 * @param np_metas ArrayMetadata
 */
void NumpyStorage::store_array_meta(const uint64_t *storage_id, ArrayMetadata *np_metas) const {
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
    uint64_t size = sizeof(uint32_t) * np_metas->dims.size();

    //plus the other metas
    size += sizeof(np_metas->elem_size) + sizeof(np_metas->inner_type) + sizeof(np_metas->partition_type);

    //allocate plus the bytes counter
    unsigned char *byte_array = (unsigned char *) malloc(size + sizeof(uint64_t));

    // Copy num bytes
    memcpy(byte_array, &size, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    //copy everything from the metas
    memcpy(byte_array + offset, &np_metas->elem_size, sizeof(np_metas->elem_size));
    offset += sizeof(np_metas->elem_size);
    memcpy(byte_array + offset, &np_metas->inner_type, sizeof(np_metas->inner_type));
    offset += sizeof(np_metas->inner_type);
    memcpy(byte_array + offset, &np_metas->partition_type, sizeof(np_metas->partition_type));
    offset += sizeof(np_metas->partition_type);
    memcpy(byte_array + offset, np_metas->dims.data(), sizeof(uint32_t) * np_metas->dims.size());


    // Copy ptr to bytearray
    memcpy(values, &byte_array, sizeof(unsigned char *));

    // Finally, we write the data
    writer->write_to_cassandra(keys, values);
}

/***
 * Write a complete numpy ndarray by using the partitioning mechanism defined in the metadata
 * @param storage_id identifying the numpy ndarray
 * @param np_metas ndarray characteristics
 * @param numpy to be saved into storage
 */
void NumpyStorage::store_entire_array(const uint64_t *storage_id, ArrayMetadata *np_metas, PyArrayObject *numpy) const {
    void *data = PyArray_BYTES(numpy);
    SpaceFillingCurve::PartitionGenerator *partitions_it = this->partitioner.make_partitions_generator(np_metas, data);

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
        writer->write_to_cassandra(keys, values);
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
const ArrayMetadata *NumpyStorage::read_array_meta(const uint64_t *storage_id, CacheTable *cache) const {
    // Get metas from Cassandra
    int32_t cluster_id = -1;

    char *buffer = (char *) malloc(sizeof(uint64_t *) + sizeof(int32_t));
    // UUID
    uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);
    c_uuid[0] = *storage_id;
    c_uuid[1] = *(storage_id + 1);
    // Copy uuid
    memcpy(buffer, &c_uuid, sizeof(uint64_t *));
    int32_t offset = sizeof(uint64_t *);
    // Cluster id
    memcpy(buffer + offset, &cluster_id, sizeof(cluster_id));
    // We fetch the data
    std::vector<const TupleRow *> results = cache->get_crow(buffer);

    if (results.empty()) throw ModuleException("Metadata for the array can't be found");

    if (results.size() != 1) throw ModuleException("Different metadata for the same array found, impossible!");

    // pos 0 is block id, pos 1 is payload
    const unsigned char *payload = *((const unsigned char **) (results[0]->get_element(1)));

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

    for (const TupleRow *&v : results) delete (v);
    return arr_metas;
}


/***
 * Reads a numpy ndarray by iterating the blocks thanks to the prefetch, guided by the token ranges
 * @param storage_id of the array to retrive
 * @param tokens Ranges defining the data to be read
 * @return Numpy ndarray as a Python object
 */
PyObject *
NumpyStorage::read_by_tokens(const uint64_t *storage_id, const std::vector<std::pair<int64_t, int64_t>> &tokens) {
    // TODO To be implemented
    throw ModuleException("To be implemented");
}


/***
 * Reads a numpy ndarray by fetching the clusters indipendently
 * @param storage_id of the array to retrieve
 * @return Numpy ndarray as a Python object
 */
PyObject *NumpyStorage::read(const uint64_t *storage_id) {

    std::vector<std::map<std::string, std::string> > keysnames = {
            {{"name", "storage_id"}},
            {{"name", "cluster_id"}}
    };
    std::vector<std::map<std::string, std::string> > colsnames = {
            {{"name", "block_id"}},
            {{"name", "payload"}}
    };

    std::map<std::string, std::string> config;
    config["cache_size"] = "0";
    config["writer_par"] = "1";
    config["writer_buffer"] = "0";

    const TableMetadata *metas = this->writer->get_metadata();

    CacheTable *cache = this->storage->make_cache(metas->get_table_name(), metas->get_keyspace(), keysnames, colsnames,
                                                  config);
    std::shared_ptr<const std::vector<ColumnMeta> > keys_metas = cache->get_metadata()->get_keys();
    uint32_t keys_size = (*--keys_metas->end()).size + (*--keys_metas->end()).position;

    const ArrayMetadata *arr_meta = read_array_meta(storage_id, cache);

    if (!arr_meta) throw ModuleException("Numpy array metadatas not present");

    std::vector<const TupleRow *> result, all_results;
    std::vector<Partition> all_partitions;

    uint64_t *c_uuid = nullptr;
    char *buffer = nullptr;
    int32_t cluster_id = 0, offset = 0;
    int32_t *block = nullptr;
    int32_t half_int = 0;//-1 >> sizeof(int32_t)/2; //TODO be done properly

    SpaceFillingCurve::PartitionGenerator *partitions_it = this->partitioner.make_partitions_generator(arr_meta,
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
        result = cache->get_crow(new TupleRow(keys_metas, keys_size, buffer));
        //build cluster
        all_results.insert(all_results.end(), result.begin(), result.end());
        for (const TupleRow *row:result) {
            block = (int32_t *) row->get_element(0);
            char **chunk = (char **) row->get_element(1);
            all_partitions.emplace_back(
                    Partition((uint32_t) cluster_id + half_int, (uint32_t) *block + half_int, *chunk));
        }
    }


    delete (cache);

    if (all_partitions.empty()) {
        throw ModuleException("no npy found on sys");
    }

    void *data = partitions_it->merge_partitions(arr_meta, all_partitions);

    for (const TupleRow *item:all_results) delete (item);
    npy_intp *dims = new npy_intp[arr_meta->dims.size()];
    for (uint32_t i = 0; i < arr_meta->dims.size(); ++i) {
        dims[i] = arr_meta->dims[i];
    }
    try {
        return PyArray_SimpleNewFromData((int32_t) arr_meta->dims.size(),
                                         dims,
                                         arr_meta->inner_type, data);
    }
    catch (std::exception e) {
        if (PyErr_Occurred()) PyErr_Print();
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

/***
 * Extract the metadatas of the given numpy ndarray and return a new ArrayMetadata with its representation
 * @param numpy Ndarray to extract the metadata
 * @return ArrayMetadata defining the information to reconstruct a numpy ndarray
 */
ArrayMetadata *NumpyStorage::get_np_metadata(PyArrayObject *numpy) const {
    int64_t ndims = PyArray_NDIM(numpy);
    npy_intp *shape = PyArray_SHAPE(numpy);

    ArrayMetadata *shape_and_type = new ArrayMetadata();
    shape_and_type->inner_type = PyArray_TYPE(numpy);

    //TODO implement as a union
    if (shape_and_type->inner_type == NPY_INT8) shape_and_type->elem_size = sizeof(int8_t);
    else if (shape_and_type->inner_type == NPY_UINT8) shape_and_type->elem_size = sizeof(uint8_t);
    else if (shape_and_type->inner_type == NPY_INT16) shape_and_type->elem_size = sizeof(int16_t);
    else if (shape_and_type->inner_type == NPY_UINT16) shape_and_type->elem_size = sizeof(uint16_t);
    else if (shape_and_type->inner_type == NPY_INT32) shape_and_type->elem_size = sizeof(int32_t);
    else if (shape_and_type->inner_type == NPY_UINT32) shape_and_type->elem_size = sizeof(uint32_t);
    else if (shape_and_type->inner_type == NPY_INT64) shape_and_type->elem_size = sizeof(int64_t);
    else if (shape_and_type->inner_type == NPY_LONGLONG) shape_and_type->elem_size = sizeof(int64_t);
    else if (shape_and_type->inner_type == NPY_UINT64) shape_and_type->elem_size = sizeof(uint64_t);
    else if (shape_and_type->inner_type == NPY_ULONGLONG) shape_and_type->elem_size = sizeof(uint64_t);
    else if (shape_and_type->inner_type == NPY_DOUBLE) shape_and_type->elem_size = sizeof(npy_double);
    else if (shape_and_type->inner_type == NPY_FLOAT16) shape_and_type->elem_size = sizeof(npy_float16);
    else if (shape_and_type->inner_type == NPY_FLOAT32) shape_and_type->elem_size = sizeof(npy_float32);
    else if (shape_and_type->inner_type == NPY_FLOAT64) shape_and_type->elem_size = sizeof(npy_float64);
    else if (shape_and_type->inner_type == NPY_FLOAT128) shape_and_type->elem_size = sizeof(npy_float128);
    else if (shape_and_type->inner_type == NPY_BOOL) shape_and_type->elem_size = sizeof(bool);
    else if (shape_and_type->inner_type == NPY_BYTE) shape_and_type->elem_size = sizeof(char);
    else if (shape_and_type->inner_type == NPY_LONG) shape_and_type->elem_size = sizeof(long);
    else if (shape_and_type->inner_type == NPY_LONGLONG) shape_and_type->elem_size = sizeof(long long);
    else if (shape_and_type->inner_type == NPY_SHORT) shape_and_type->elem_size = sizeof(short);
    else throw ModuleException("Numpy data type still not supported");

    // Copy elements per dimension
    shape_and_type->dims = std::vector<uint32_t>((uint64_t) ndims);//PyArray_SHAPE()
    for (int32_t dim = 0; dim < ndims; ++dim) {
        shape_and_type->dims[dim] = (uint32_t) shape[dim];
    }
    return shape_and_type;
}
