#include "NumpyStorage.h"


NumpyStorage::NumpyStorage(std::shared_ptr<StorageInterface> storage, ArrayPartitioner &algorithm) {
    this->storage = storage;
    this->partitioner = algorithm;
}

ArrayMetadata NumpyStorage::store(std::string table, std::string keyspace, std::string attr_name, CassUuid &storage_id, PyArrayObject* numpy) {
    ArrayMetadata np_metas = get_np_metadata(numpy);
    void* data = PyArray_BYTES(numpy);
    std::vector<Partition> parts = partitioner.make_partitions(np_metas,data); //z-order or whatever we want
    std::vector< std::map<std::string,std::string> > keysnames = {{{"name", "partid"}},{{"name","time"}}};
    std::vector< std::map<std::string,std::string> > colsnames = { {{"name", "x"}},{{"name", "y"}},{{"name", "z"}}};

    std::map <std::string, std::string> config;
    config["writer_par"] = "4";
    config["writer_buffer"] = "20";
    config["cache_size"] = "10";
    Writer* W = this->storage->make_writer(table.c_str(),keyspace.c_str(),keysnames,colsnames,config);

    char *keys = nullptr;
    for (uint32_t npart = 0; npart<parts.size(); ++npart) {
        keys = (char*) malloc(sizeof(int32_t)*2);
  /*      memcpy(keys,&uuid,sizeof(uuid));
        memcpy(keys,&attr_name.c_str(),sizeof(char*));
        memcpy(keys,&parts[npart].block_id,sizeof(int32_t));
        memcpy(keys,&parts[npart].cluster_id,sizeof(int32_t));
        W->write_to_cassandra(keys,parts[npart].data);*/
    }

    delete(W); //TODO we don't want to get stuck here, writing should proceed async
    return np_metas;

}

PyObject* NumpyStorage::read(std::string table, TupleRow* keys, ArrayMetadata &np){
//TODO To be implemented
    return nullptr;
}

ArrayMetadata NumpyStorage::get_np_metadata(PyArrayObject *numpy) {
    ArrayMetadata shape_and_type = ArrayMetadata();
    shape_and_type.inner_type = PyArray_TYPE(numpy);
    shape_and_type.dims=std::vector<int32_t>();//PyArray_SHAPE()
    return shape_and_type;
}
