#include "NumpyStorage.h"


NumpyStorage::NumpyStorage(std::string table, std::string keyspace, std::shared_ptr<StorageInterface> storage) {

    std::vector<std::map<std::string, std::string> > keysnames = {
            {{"name", "storage_id"}},
            {{"name", "cluster_id"}},
            {{"name", "block_id"}}
    };
    std::vector<std::map<std::string, std::string> > colsnames = {
            {{"name", "payload"}}
    };

    std::map<std::string, std::string> config;

    this->storage = storage;
    this->writer = this->storage->make_writer(table.c_str(), keyspace.c_str(), keysnames, colsnames, config);
}


NumpyStorage::~NumpyStorage() {
    if (writer) {
        delete (writer->get_metadata());
        delete (writer);
    }
};


const ArrayMetadata *
NumpyStorage::store(const CassUuid &storage_id, PyArrayObject *numpy) const {
    ArrayMetadata *np_metas = get_np_metadata(numpy);
    np_metas->partition_type = ZORDER_ALGORITHM;
    void *data = PyArray_BYTES(numpy);
    SpaceFillingCurve::PartitionGenerator* partitions_it = this->partitioner.make_partitions_generator(np_metas, data);

    char *keys = nullptr;
    void* values = nullptr;
    uint32_t offset = 0, keys_size = sizeof(uint64_t *) + sizeof(int32_t) * 2;
    uint64_t *c_uuid = nullptr;
    uint32_t half_int = 0;//(uint32_t)-1 >> (sizeof(uint32_t)*CHAR_BIT/2); //TODO be done properly
    int32_t cluster_id, block_id;
    while (!partitions_it->isDone()) {
        Partition part = partitions_it->getNextPartition();
        keys = (char *) malloc(keys_size);
        //UUID
        c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);//new uint64_t[2];
        c_uuid[0] = storage_id.time_and_version;
        c_uuid[1] = storage_id.clock_seq_and_node;
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
    delete(partitions_it);
    return np_metas;
}


PyObject *NumpyStorage::read(std::string table, std::string keyspace, const CassUuid &storage_id,
                             const ArrayMetadata *arr_meta) {

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

    CacheTable *cache = this->storage->make_cache(table.c_str(), keyspace.c_str(), keysnames, colsnames, config);
    std::shared_ptr<const std::vector<ColumnMeta> > keys_metas = cache->get_metadata()->get_keys();
    uint32_t keys_size = (*--keys_metas->end()).size + (*--keys_metas->end()).position;


    std::vector<const TupleRow *> result, all_results;
    std::vector<Partition> all_partitions;

    uint64_t *c_uuid = nullptr;
    char *buffer = nullptr;
    int32_t cluster_id = 0, offset = 0;
    int32_t *block = nullptr;
    int32_t half_int = 0;//-1 >> sizeof(int32_t)/2; //TODO be done properly

    SpaceFillingCurve::PartitionGenerator* partitions_it = this->partitioner.make_partitions_generator(arr_meta,
                                                                                                        nullptr);

    while (!partitions_it->isDone()) {
        cluster_id = partitions_it->computeNextClusterId();
        buffer = (char *) malloc(keys_size);
        //UUID
        c_uuid = new uint64_t[2];
        c_uuid[0] = storage_id.time_and_version;
        c_uuid[1] = storage_id.clock_seq_and_node;
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
            all_partitions.emplace_back(Partition((uint32_t) cluster_id+half_int, (uint32_t) *block+half_int,*chunk));
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
    shape_and_type->dims = std::vector<uint32_t>((uint64_t) ndims);//PyArray_SHAPE()
    for (int32_t dim = 0; dim < ndims; ++dim) {
        shape_and_type->dims[dim] = (uint32_t) shape[dim];
    }
    return shape_and_type;
}
