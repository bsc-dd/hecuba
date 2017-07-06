#include "NumpyStorage.h"


NumpyStorage::NumpyStorage(std::shared_ptr<StorageInterface> storage, ArrayPartitioner &algorithm) {
    this->storage = storage;
    this->partitioner = algorithm;
}

ArrayMetadata* NumpyStorage::store(std::string table, std::string keyspace, std::string attr_name, const CassUuid &storage_id, PyArrayObject* numpy) const {
    ArrayMetadata *np_metas = get_np_metadata(numpy);
    void* data = PyArray_BYTES(numpy);
    std::vector<Partition> parts = partitioner.make_partitions(np_metas,data); //z-order or whatever we want
    std::vector< std::map<std::string,std::string> > keysnames = {
            {{"name", "storage_id"}},{{"name","attr_name"}},
            {{"name", "cluster_id"}},{{"name","block_id"}}
    };
    std::vector< std::map<std::string,std::string> > colsnames = {
            {{"name", "payload"},{"free","false"}}
    };

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";

    Writer* W = this->storage->make_writer(table.c_str(),keyspace.c_str(),keysnames,colsnames,config);

    char *keys, *values = nullptr;
    uint32_t keys_size = sizeof(uint64_t*)+sizeof(char*)+sizeof(int32_t)*2;
    for (uint32_t npart = 0; npart<parts.size(); ++npart) {
        keys = (char*) malloc(keys_size);
        //UUID
        uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);
        uint32_t offset = 0;
        *c_uuid = storage_id.time_and_version;
        *(c_uuid + 1) = storage_id.clock_seq_and_node;
        memcpy(keys, &c_uuid, sizeof(uint64_t*));

        offset += sizeof(uint64_t*);

        //ATTR NAME
        char *attr_name_c = strdup(attr_name.c_str());
        memcpy(keys+offset,&attr_name_c,sizeof(char*));
        offset += sizeof(char *);
        //Cluster id
        memcpy(keys+offset,&parts[npart].cluster_id,sizeof(int32_t));
        offset += sizeof(int32_t);
        //Block id
        memcpy(keys+offset,&parts[npart].block_id,sizeof(int32_t));
        //COPY VALUES

        values = (char*) malloc (sizeof(char*));
        memcpy(values,&parts[npart].data,sizeof(char*));
        //FINALLY WE WRITE THE DATA
        W->write_to_cassandra(keys,values);
    }
    delete(W); //TODO we don't want to get stuck here, writing should proceed async
    return np_metas;

}

PyObject* NumpyStorage::read(std::string table, TupleRow* keys, ArrayMetadata &np){
//TODO To be implemented
    return nullptr;
}

ArrayMetadata* NumpyStorage::get_np_metadata(PyArrayObject *numpy) const {
    int64_t ndims = PyArray_NDIM(numpy);
    npy_intp *shape = PyArray_SHAPE(numpy);

    ArrayMetadata *shape_and_type = new ArrayMetadata();
    shape_and_type->inner_type = PyArray_TYPE(numpy);
    shape_and_type->dims=std::vector<int32_t>((uint64_t)ndims);//PyArray_SHAPE()
    for (int32_t dim = 0; dim<ndims; ++dim){
        shape_and_type->dims[dim]=(int32_t) shape[dim];
    }
    return shape_and_type;
}
