#include "ArrayDataStore.h"


ArrayDataStore::ArrayDataStore(const char *table, const char *keyspace, CassSession *session,
                               std::map<std::string, std::string> &config) {
    char * full_name=(char *)malloc(strlen(table)+strlen(keyspace)+ 2);
    sprintf(full_name,"%s.%s",keyspace,table);
    this->TN = std::string(full_name); //lgarrobe

    std::vector<std::map<std::string, std::string> > keys_names = {{{"name", "storage_id"}},
                                                                   {{"name", "cluster_id"}},
                                                                   {{"name", "block_id"}}};

    std::vector<std::map<std::string, std::string> > columns_names = {{{"name", "payload"}}};


    TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
    this->cache = new CacheTable(table_meta, session, config);

	// Arrow tables:
	//    - Writing to arrow uses temporal 'buffer' table
	//    - Reading from arrow uses 'arrow' table
	// Both have the SAME keys but DIFFERENT values!
    std::vector<std::map<std::string, std::string> > keys_arrow_names = {{{"name", "storage_id"}},
                                                                         {{"name", "col_id"}}};

	// Temporal buffer for writing to arrow
    std::vector<std::map<std::string, std::string> > columns_buffer_names = {{{"name", "row_id"}},
                                                                            {{"name", "size_elem"}},
                                                                            {{"name", "payload"}}};

	// Arrow table (for reading)
    std::vector<std::map<std::string, std::string> > columns_arrow_names = {{{"name", "arrow_addr"}},
                                                                            {{"name", "arrow_size"}}};
	// Create table names: table_buffer, table_arrow, keyspace_arrow
    std::string table_buffer = table;
    table_buffer.append("_buffer");
    std::string table_arrow = table;
    table_arrow.append("_arrow");
    std::string keyspace_arrow = keyspace;
    keyspace_arrow.append("_arrow");

	// Prepare cache for WRITE
    TableMetadata *table_meta_arrow_write = new TableMetadata(table_buffer.c_str(), keyspace_arrow.c_str(),
                                                        keys_arrow_names, columns_buffer_names, session);
    this->cache_arrow_write = new CacheTable(table_meta_arrow_write, session, config); // FIXME can be removed?

	// Prepare cache for READ
    TableMetadata *table_meta_arrow = new TableMetadata(table_arrow.c_str(), keyspace_arrow.c_str(),
                                                        keys_arrow_names, columns_arrow_names, session);
    this->cache_arrow = new CacheTable(table_meta_arrow, session, config);

    std::vector<std::map<std::string, std::string>> read_keys_names(keys_names.begin(), (keys_names.end() - 1));
    std::vector<std::map<std::string, std::string>> read_columns_names = columns_names;
    read_columns_names.insert(read_columns_names.begin(), keys_names.back());

    table_meta = new TableMetadata(table, keyspace, read_keys_names, read_columns_names, session);
    this->read_cache = new CacheTable(table_meta, session, config);

    //Metadata needed only for *reading* numpy metas from hecuba.istorage
    //lgarrobe
    std::vector<std::map<std::string, std::string> > metadata_keys = {{{"name", "storage_id"}}};

    std::vector<std::map<std::string, std::string> > metadata_columns = {{{"name", "base_numpy"}}
									,{{"name", "class_name"}}
									,{{"name", "name"}}
									,{{"name", "numpy_meta"}}};

    TableMetadata *metadata_table_meta = new TableMetadata("istorage", "hecuba", metadata_keys, metadata_columns, session);

    this->metadata_cache = new CacheTable(metadata_table_meta, session, config);


    std::vector<std::map<std::string, std::string>> read_metadata_keys  (metadata_keys.begin(), (metadata_keys.end() ));
    std::vector<std::map<std::string, std::string>> read_metadata_columns = metadata_columns;

    metadata_table_meta = new TableMetadata("istorage", "hecuba", read_metadata_keys, read_metadata_columns, session);

    this->metadata_read_cache = new CacheTable(metadata_table_meta, session, config);

}


ArrayDataStore::~ArrayDataStore() {
    delete (this->cache);
    delete (this->read_cache);
    delete (this->metadata_cache);
    delete (this->metadata_read_cache);
};

/***
 * Stores the array metadata by setting the cluster and block ids to -1. Deletes the array metadata afterwards.
 * @param storage_id UUID used as part of the key
 * @param np_metas ArrayMetadata
 */

/*
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
*/

/***
 * Write a complete numpy ndarray by using the partitioning mechanism defined in the metadata
 * @param storage_id identifying the numpy ndarray
 * @param part partition to store
 */
void ArrayDataStore::store_numpy_partition_into_cas(const uint64_t *storage_id , Partition part) const {
    char *keys = nullptr;
    void *values = nullptr;
    uint32_t half_int = 0;//(uint32_t)-1 >> (sizeof(uint32_t)*CHAR_BIT/2); //TODO be done properly
    uint32_t offset = 0, keys_size = sizeof(uint64_t *) + sizeof(int32_t) * 2;
    int32_t cluster_id, block_id;
    uint64_t *c_uuid = nullptr;

        keys = (char *) malloc(keys_size);
        //UUID
        c_uuid = new uint64_t[2]{*storage_id, *(storage_id + 1)};
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


/***
 * Write a complete numpy ndarray by columns (using arrow)
 * @param storage_id identifying the numpy ndarray
 * @param np_metas ndarray characteristics
 * @param numpy to be saved into storage
 */
void ArrayDataStore::store_numpy_into_cas_as_arrow(const uint64_t *storage_id,
									ArrayMetadata &metadata, void *data) const {

	assert( metadata.dims.size() <= 2 ); // First version only supports 2 dimensions

	// Calculate row and element sizes
	uint64_t row_size	= metadata.strides[0];
	uint32_t elem_size	= metadata.elem_size;

	// Calculate number of rows and columns
	uint64_t num_columns = metadata.dims[1];
	uint64_t num_rows	= metadata.dims[0];

    //arrow
	arrow::Status status;
    auto memory_pool = arrow::default_memory_pool(); //arrow
    auto field = arrow::field("field", arrow::binary());
    std::vector<std::shared_ptr<arrow::Field>> fields = {field};
    auto schema = std::make_shared<arrow::Schema>(fields);

    for(uint64_t i = 0; i < num_columns; ++i) {
        arrow::BinaryBuilder builder(arrow::binary(), memory_pool); //arrow
        status = builder.Resize(num_rows); //arrow
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at builder.Resize" << std::endl;
        for (uint64_t j = 0; j < num_rows; ++j) {
            const char * src = (char*)data + j*row_size + i*elem_size; // data[j][i]
            status = builder.Append(src, elem_size);
            if (!status.ok())
                std::cout << "Status: " << status.ToString() << " at builder.Append" << std::endl;
        }
        std::shared_ptr<arrow::Array> array;
        status = builder.Finish(&array);
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at builder.Finish" << std::endl;
        auto batch = arrow::RecordBatch::Make(schema, num_rows, {array});
        std::shared_ptr<arrow::io::BufferOutputStream> bufferOutputStream;
        status = arrow::io::BufferOutputStream::Create(0, memory_pool, &bufferOutputStream);
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at BufferOutputStream::Create" << std::endl;
        std::shared_ptr<arrow::ipc::RecordBatchWriter> file_writer;
        status = arrow::ipc::RecordBatchFileWriter::Open(bufferOutputStream.get(), schema, &file_writer);
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at RecordBatchFileWriter::Open" << std::endl;

        status = file_writer->WriteRecordBatch(*batch);
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at file_writer->WriteRecordBatch" << std::endl;
        status = file_writer->Close();
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at file_writer->Close" << std::endl;
        status = bufferOutputStream->Close();
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at bufferOutputStream->Close" << std::endl;

        std::shared_ptr<arrow::Buffer> result;
        status = bufferOutputStream->Finish(&result); //arrow
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at bufferOutputStream->Finish" << std::endl;

        //Store Column
        // Allocate memory for keys
        char* _keys = (char *) malloc(sizeof(uint64_t*) + sizeof(uint64_t));
        char* _k = _keys;
        // Copy data
        //UUID
        uint64_t *c_uuid  = new uint64_t[2]{*storage_id, *(storage_id + 1)};
        // [0] = storage_id.time_and_version;
        // [1] = storage_id.clock_seq_and_node;
        memcpy(_k, &c_uuid, sizeof(uint64_t*)); //storage_id
        _k += sizeof(uint64_t*);
        memcpy(_k, &i, sizeof(uint64_t)); //col_id

        // Allocate memory for values
        uint32_t values_size = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(char*);
        char* _values = (char*) malloc(values_size);
        char* _val = _values; // Temporal to avoid modifications on the original value
        // Copy data
        uint64_t row_id = 0;
        memcpy(_val, &row_id, sizeof(uint64_t));	//row_id
        _val += sizeof(uint64_t);
        memcpy(_val, &elem_size, sizeof(uint32_t)); //elem_size
        _val += sizeof(uint32_t);

        uint64_t arrow_size = result->size();
        char *mypayload = (char *)malloc(sizeof(uint64_t) + arrow_size);
        memcpy(mypayload, &arrow_size, sizeof(uint64_t));
        memcpy(mypayload + sizeof(uint64_t), result->data(), arrow_size);

        memcpy(_val, &mypayload, sizeof(char*));	//payload

        cache_arrow_write->put_crow( (void*)_keys, (void*)_values ); //Send column to cassandra
    }
}

/***
 * Write a complete numpy ndarray by using the partitioning mechanism defined in the metadata
 * @param storage_id identifying the numpy ndarray
 * @param np_metas ndarray characteristics
 * @param numpy to be saved into storage
 */
void ArrayDataStore::store_numpy_into_cas(const uint64_t *storage_id, ArrayMetadata &metadata, void *data) const {

    SpaceFillingCurve::PartitionGenerator *partitions_it = this->partitioner.make_partitions_generator(metadata, data);

    while (!partitions_it->isDone()) {
        Partition part = partitions_it->getNextPartition();
        store_numpy_partition_into_cas(storage_id, part);
    }
    //this->partitioner.serialize_metas();
    delete (partitions_it);

    // No need to flush the elements because the metadata are written after the data thanks to the queue
    store_numpy_into_cas_as_arrow(storage_id, metadata, data);
}

void ArrayDataStore::store_numpy_into_cas_by_coords(const uint64_t *storage_id, ArrayMetadata &metadata, void *data,
                                                    std::list<std::vector<uint32_t> > &coord) const {

    SpaceFillingCurve::PartitionGenerator *partitions_it = SpaceFillingCurve::make_partitions_generator(metadata, data,
                                                                                                        coord);

    std::set<int32_t> clusters = {};
    std::list<Partition> partitions = {};

    while (!partitions_it->isDone()) { clusters.insert(partitions_it->computeNextClusterId()); }
    partitions_it = new ZorderCurveGeneratorFiltered(metadata, data, coord);
    while (!partitions_it->isDone()) {
        clusters.insert(partitions_it->computeNextClusterId());
        auto part = partitions_it->getNextPartition();
        if (clusters.find(part.cluster_id) != clusters.end()) partitions.push_back(part);
    }

    for (auto it = partitions.begin(); it != partitions.end(); ++it) {
        auto part = *it;
        store_numpy_partition_into_cas(storage_id, part);
    }
    //this->partitioner.serialize_metas();
    delete (partitions_it);
    // FIXME store_numpy_into_cas_by_coords_as_arrow(storage_id, metadata, data, coord);
}

/***
 * Reads the metadata from the storage as an ArrayMetadata for latter use
 * @param storage_id identifing the numpy ndarray
 * @param cache
 * @return
 */
ArrayMetadata *ArrayDataStore::read_metadata(const uint64_t *storage_id) const {
    // Get metas from Cassandra
    //int32_t cluster_id = -1, block_id = -1;
    //char *buffer = (char *) malloc(sizeof(uint64_t *) + sizeof(int32_t) * 2);
    char *buffer = (char *) malloc(sizeof(uint64_t *));
    // UUID
    uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);
    c_uuid[0] = *storage_id;
    c_uuid[1] = *(storage_id + 1);
    // Copy uuid
    memcpy(buffer, &c_uuid, sizeof(uint64_t *));
   
    //int32_t offset = sizeof(uint64_t *);
    // Cluster id
    //memcpy(buffer + offset, &cluster_id, sizeof(int32_t));
    //offset += sizeof(int32_t);
    // Cluster id
    //memcpy(buffer + offset, &block_id, sizeof(int32_t));

    // We fetch the data
    std::vector<const TupleRow *> results = metadata_cache->get_crow(buffer);

    if (results.empty()) throw ModuleException("Metadata for the array can't be found");
    // pos 0 is block id, pos 1 is payload
    const unsigned char *payload = *((const unsigned char **) (results[0]->get_element(0))); //lgarrobe canvio el 1 per un 0

    const uint64_t num_bytes = *((uint64_t *) payload);
    // Move pointer to the beginning of the data
    payload = payload + sizeof(num_bytes);


    uint32_t bytes_offset = 0;
    ArrayMetadata *arr_metas = new ArrayMetadata();
    // Load data

    memcpy(&arr_metas->flags, payload + bytes_offset, sizeof(arr_metas->flags));
    bytes_offset += sizeof(arr_metas->flags);
    memcpy(&arr_metas->elem_size, payload + bytes_offset, sizeof(arr_metas->elem_size));
    bytes_offset += sizeof(arr_metas->elem_size);
    memcpy(&arr_metas->partition_type, payload + bytes_offset, sizeof(arr_metas->partition_type));
    bytes_offset += sizeof(arr_metas->partition_type);
    memcpy(&arr_metas->typekind, payload + bytes_offset, sizeof(arr_metas->typekind));
    bytes_offset += sizeof(arr_metas->typekind);
    memcpy(&arr_metas->byteorder, payload + bytes_offset, sizeof(arr_metas->byteorder));
    bytes_offset += sizeof(arr_metas->byteorder);
    uint64_t nbytes = (num_bytes - bytes_offset)/2;
    uint32_t nelem = (uint32_t) nbytes / sizeof(uint32_t);
    if (nbytes % sizeof(uint32_t) != 0) throw ModuleException("something went wrong reading the dims of a numpy");
    arr_metas->dims = std::vector<uint32_t>(nelem);
    memcpy(arr_metas->dims.data(), payload + bytes_offset, nbytes);
    bytes_offset += nbytes;
    arr_metas->strides = std::vector<uint32_t>(nelem);
    memcpy(arr_metas->strides.data(), payload + bytes_offset, nbytes);

    for (const TupleRow *&v : results) delete (v);
    return arr_metas;
}


/***
 * Reads a numpy ndarray by fetching the clusters indipendently
 * @param storage_id of the array to retrieve
 * @return Numpy ndarray as a Python object
 */
void ArrayDataStore::read_numpy_from_cas(const uint64_t *storage_id, ArrayMetadata &metadata, void *save) {

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
    this->cache->flush_elements();
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
        TupleRow *block_key = new TupleRow(keys_metas, keys_size, buffer);
        result = read_cache->get_crow(block_key);
        delete (block_key);
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
    partitions_it->merge_partitions(metadata, all_partitions, save);
    for (const TupleRow *item:all_results) delete (item);
    delete (partitions_it);
}

void ArrayDataStore::read_numpy_from_cas_by_coords(const uint64_t *storage_id, ArrayMetadata &metadata,
                                                   std::list<std::vector<uint32_t> > &coord, void *save) {
    std::shared_ptr<const std::vector<ColumnMeta> > keys_metas = read_cache->get_metadata()->get_keys();
    uint32_t keys_size = (*--keys_metas->end()).size + (*--keys_metas->end()).position;
    std::vector<const TupleRow *> result, all_results;
    std::vector<Partition> all_partitions;
    uint64_t *c_uuid = nullptr;
    char *buffer = nullptr;
    int32_t offset = 0;
    int32_t *block = nullptr;
    int32_t half_int = 0;//-1 >> sizeof(int32_t)/2; //TODO be done properly

    SpaceFillingCurve::PartitionGenerator *partitions_it = SpaceFillingCurve::make_partitions_generator(metadata,
                                                                                                        nullptr, coord);
    std::set<int32_t> clusters = {};

    while (!partitions_it->isDone()) {
        clusters.insert(partitions_it->computeNextClusterId());
    }

    std::set<int32_t>::iterator it = clusters.begin();
    for (; it != clusters.end(); ++it) {
        buffer = (char *) malloc(keys_size);
        //UUID
        c_uuid = new uint64_t[2]{*storage_id, *(storage_id + 1)};
        //[0] time_and_version;
        //[1] clock_seq_and_node;
        memcpy(buffer, &c_uuid, sizeof(uint64_t *));
        offset = sizeof(uint64_t *);
        //Cluster id
        memcpy(buffer + offset, &(*it), sizeof(*it));
        //We fetch the data
        TupleRow *block_key = new TupleRow(keys_metas, keys_size, buffer);
        result = read_cache->get_crow(block_key);
        delete (block_key);
        //build cluster
        all_results.insert(all_results.end(), result.begin(), result.end());
        for (const TupleRow *row:result) {
            block = (int32_t *) row->get_element(0);
            char **chunk = (char **) row->get_element(1);
            all_partitions.emplace_back(
                    Partition((uint32_t) *it + half_int, (uint32_t) *block + half_int, *chunk));
        }
    }

    if (all_partitions.empty()) {
        throw ModuleException("no npy found on sys");
    }
    partitions_it->merge_partitions(metadata, all_partitions, save);
    for (const TupleRow *item:all_results) delete (item);
    delete (partitions_it);
}

void ArrayDataStore::read_numpy_from_cas_arrow(const uint64_t *storage_id, ArrayMetadata &metadata,
                                                   std::vector<uint32_t> &cols, void *save) {
    std::shared_ptr<const std::vector<ColumnMeta> > keys_metas = cache_arrow->get_metadata()->get_keys();
    uint32_t keys_size = (*--keys_metas->end()).size + (*--keys_metas->end()).position;
    std::vector<const TupleRow *> result, all_results;
    uint64_t *c_uuid = nullptr;
    char *_keys = nullptr;
    int32_t offset = 0;
    int32_t *block = nullptr;

    for (uint32_t it = 0; it < cols.size(); ++it) {
        _keys = (char *) malloc(keys_size);
        //UUID
        c_uuid = new uint64_t[2]{*storage_id, *(storage_id + 1)};
        //[0] time_and_version;
        //[1] clock_seq_and_node;
        memcpy(_keys, &c_uuid, sizeof(uint64_t *));
        offset = sizeof(uint64_t *);
        //col id
        memcpy(_keys + offset, &cols[it], sizeof(cols[it]));

        //We fetch the data
        TupleRow *block_key = new TupleRow(keys_metas, keys_size, _keys);

        result = cache_arrow->get_crow(block_key);// FIXME use Yolanda's IN instead of a call to cassandra per column

        delete (block_key);
        free(_keys);    // FIXME  This should be done everywhere in this file
        delete(c_uuid); // FIXME  This should be done everywhere in this file
    }

    //partitions_it->merge_partitions(metadata, all_partitions, save);
    //for (const TupleRow *item:all_results) delete (item);
    //delete (partitions_it);
}
