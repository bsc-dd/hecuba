#include "ArrayDataStore.h"

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

#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>


#include "murmur3.hpp"
#include <endian.h>

#include <climits>
#include <list>
#include <set>

#include <algorithm>
#include <cctype>
#include <string>

#define PMEM_OFFSET 8
#define MAX_RETRIES 5

ArrayDataStore::ArrayDataStore(const char *table, const char *keyspace, CassSession *session,
                               std::map<std::string, std::string> &config) {

    char * env_path = std::getenv("HECUBA_ARROW");
    if (env_path != nullptr) {
        std::string hecuba_arrow(env_path);
        std::transform(hecuba_arrow.begin(), hecuba_arrow.end(), hecuba_arrow.begin(),
                        [](unsigned char c){ return std::tolower(c); });
        if ((hecuba_arrow.compare("no")==0) ||
            (hecuba_arrow.compare("false")==0) ) {
            arrow_enabled = false;
        } else {
            arrow_enabled = true;
        }
    }
    env_path = std::getenv("HECUBA_ARROW_PATH");
    if (env_path != nullptr) {
        std::string hecuba_arrow(env_path);
        arrow_path = hecuba_arrow;
    }
    env_path = std::getenv("HECUBA_ARROW_OPTANE");
    if (env_path != nullptr) {
        std::string hecuba_arrow(env_path);
        std::transform(hecuba_arrow.begin(), hecuba_arrow.end(), hecuba_arrow.begin(),
                        [](unsigned char c){ return std::tolower(c); });
        if (hecuba_arrow.compare("true")==0) {
            arrow_optane = true;
        }
    }

    char * full_name=(char *)malloc(strlen(table)+strlen(keyspace)+ 2);
    sprintf(full_name,"%s.%s",keyspace,table);
    this->TN = std::string(full_name); //lgarrobe

    std::string table_name (table);
    if (table_name.find("_arrow", table_name.length()-6) == std::string::npos) {// != COLUMNAR
    //	std::cout<< " JJ ArrayDataStore::ArrayDataStore table=" << table << std::endl;
        std::vector<std::map<std::string, std::string> > keys_names = {{{"name", "storage_id"}},
                                                                       {{"name", "cluster_id"}},
                                                                       {{"name", "block_id"}}};

        std::vector<std::map<std::string, std::string> > columns_names = {{{"name", "payload"}}};


        TableMetadata *table_meta = new TableMetadata(table, keyspace, keys_names, columns_names, session);
        this->cache = new CacheTable(table_meta, session, config);
        this->cache->get_writer()->enable_lazy_write();

        this->read_cache = this->cache; // TODO Remove 'read_cache' references

    } else { // COLUMNAR access...
        if (!arrow_enabled){
            std::cerr<< "Creating ArrayDataStore for " << full_name << ", but HECUBA_ARROW is not enabled" << std::endl;
            return;
        }
        // Arrow tables:
        //    - Writing to arrow uses temporal 'buffer' table
        //    - Reading from arrow uses 'arrow' table
        // Both have the SAME keys but DIFFERENT values!
        // FIXME  These tables should be the same (or at least not visible from here)
        std::vector<std::map<std::string, std::string> > keys_arrow_names = {{{"name", "storage_id"}},
                                                                             {{"name", "cluster_id"}},
                                                                             {{"name", "col_id"}}};

        // Temporal buffer for writing to arrow
        std::vector<std::map<std::string, std::string> > columns_buffer_names = {{{"name", "row_id"}},
                                                                                 {{"name", "size_elem"}},
                                                                                 {{"name", "payload"}}};

        // Arrow table (for reading)
        std::vector<std::map<std::string, std::string> > columns_arrow_names = {{{"name", "arrow_addr"}},
                                                                                {{"name", "arrow_size"}}};
    //	std::cout<< " JJ ArrayDataStore::ArrayDataStore ARROW " << table << std::endl;
        // Create table names: table_buffer, table_arrow (must match 'hnumpy.py' names)
        std::string table_buffer (table);
        table_buffer.replace(table_buffer.length()-6, 7, "_buffer"); // Change '_arrow' to '_buffer'

        // Prepare cache for WRITE
        TableMetadata *table_meta_arrow_write = new TableMetadata(table_buffer.c_str(), keyspace,
                                                                  keys_arrow_names, columns_buffer_names, session);
        this->cache = new CacheTable(table_meta_arrow_write, session, config); // FIXME can be removed?
        //this->cache->get_writer()->enable_lazy_write();

        // Prepare cache for READ
        TableMetadata *table_meta_arrow = new TableMetadata(table_name.c_str(), keyspace,
                                                            keys_arrow_names, columns_arrow_names, session);
        this->read_cache = new CacheTable(table_meta_arrow, session, config);
    }

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

ArrayDataStore::ArrayDataStore(const char *table, const char *keyspace, std::shared_ptr<StorageInterface> storage,
                               std::map<std::string, std::string> &config) :
    ArrayDataStore(table, keyspace, storage->get_session(), config) {
    this->storage = storage;
}


ArrayDataStore::~ArrayDataStore() {
    delete (this->cache);
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

/* Generate a token for composite key (storage_id, cluster_id)
 * Args:
 *      storage_id: Points to a UUID
 *      cluster_id: Contains a cluster_id
 * Returns an int64_t with the token that Cassandra assigns to the composite key
 */
int64_t murmur(const uint64_t *storage_id, const uint32_t cluster_id) {
        char mykey[256];
        uint64_t sidBE_high;
        uint64_t sidBE_low;
        uint32_t cidBE;
        uint16_t sidlenBE = sizeof(uint64_t)*2;
        uint16_t cidlenBE = sizeof(uint32_t);
        if (htonl(0x12345678) == 0x12345678) { //BIGENDIAN
            sidBE_high = *storage_id;
            sidBE_low  = *(storage_id+1);
            cidBE =  cluster_id;
        } else {  // LittleEndian // --> Transform to BigEndian needed.
            sidBE_high = *storage_id;
            sidBE_low  = *(storage_id+1);

            cidBE = htobe32(cluster_id);
            sidlenBE = htobe16(sidlenBE);
            cidlenBE = htobe16(cidlenBE);
        }
        char* p = mykey;
        memcpy(p, &sidlenBE, sizeof(sidlenBE));
        p+= sizeof(sidlenBE);
        memcpy(p, &sidBE_high, sizeof(sidBE_high));
        p+= sizeof(sidBE_high);
        memcpy(p, &sidBE_low, sizeof(sidBE_low));
        p+= sizeof(sidBE_low);
        memcpy(p, "", 1); // The byte '0'
        p+= 1;

        memcpy(p, &cidlenBE, sizeof(cidlenBE));
        p+= sizeof(cidlenBE);
        memcpy(p, &cidBE, sizeof(cidBE));
        p+= sizeof(cidBE);
        memcpy(p, "", 1); // The byte '0'
        p+= 1;

        int64_t token =  datastax::internal::MurmurHash3_x64_128(mykey, (int)(p-mykey), 0);

        //std::cout<< "murmur storage_id="<<std::hex<<*storage_id<<" cluster_id="<<std::dec<<cluster_id<<"-->"<<token<<std::endl;
        return token;
}
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
    //// Create token from storage_id+cluster_id (partition key)
    //int64_t token = murmur(c_uuid, cluster_id);
    //std::cout<< "JCOSTA storing partition sid="<<std::hex<<*c_uuid<<"-cid="<<std::dec<<cluster_id<<"-->"<<token<<std::endl;

        values = (char *) malloc(sizeof(char *));
        memcpy(values, &part.data, sizeof(char *));

        //FINALLY WE WRITE THE DATA
        cache->put_crow(keys, values);
}

void ArrayDataStore::wait_stores(void) const {
    cache->wait_elements();
}

CacheTable* ArrayDataStore::getWriteCache(void) const {
    return cache;
}

/* get_row_elements - Calculate #elements per dimension 
 * FIXME This code MUST BE equal to the one in NumpyStorage (No questions please, I cried also) and on SpaceFilling.cpp
 */
uint32_t ArrayDataStore::get_row_elements(ArrayMetadata &metadata) const {
    uint32_t ndims = (uint32_t) metadata.dims.size();
    uint64_t block_size = BLOCK_SIZE - (BLOCK_SIZE % metadata.elem_size);
    uint32_t row_elements = (uint32_t) std::floor(pow(block_size / metadata.elem_size, (1.0 / ndims)));
    return row_elements;
}

/***
 * Write a complete numpy ndarray by columns (using arrow)
 * @param storage_id identifying the numpy ndarray
 * @param np_metas ndarray characteristics
 * @param numpy to be saved into storage (columns are consecutive in memory)
 */
void ArrayDataStore::store_numpy_into_cas_as_arrow(const uint64_t *storage_id,
                                                   ArrayMetadata &metadata, void *data) const {

    if (!arrow_enabled) {
        std::cerr<< "store_numpy_into_cas_as_arrow called, but HECUBA_ARROW is not enabled" << std::endl;
        return;
    }

    assert( metadata.dims.size() <= 2 ); // First version only supports 2 dimensions

    // Calculate row and element sizes
    //uint64_t row_size   = metadata.strides[1];
    uint32_t elem_size  = metadata.elem_size;

    // Calculate number of rows and columns
    uint64_t num_columns = metadata.dims[1];
    uint64_t num_rows    = metadata.dims[0];

    // FIXME Encapsulate the following code into a function f(data, columns) -> Arrow
    //arrow
    arrow::Status status;
    auto memory_pool = arrow::default_memory_pool(); //arrow
    auto field = arrow::field("field", arrow::binary());
    std::vector<std::shared_ptr<arrow::Field>> fields = {field};
    auto schema = std::make_shared<arrow::Schema>(fields);

    uint32_t cluster_id = 0;
    uint32_t row_elements = get_row_elements(metadata);
    //std::cout<< "store_numpy_into_cas_as_arrow cols="<<num_columns<<", rows="<<num_rows<<", row_elements="<<row_elements<<std::endl;

    char * src = (char*)data;
    for(uint64_t i = 0; i < num_columns; ++i) {
        arrow::BinaryBuilder builder(arrow::binary(), memory_pool); //arrow
        status = builder.Resize(num_rows); //arrow
        if (!status.ok())
            std::cout << "Status: " << status.ToString() << " at builder.Resize" << std::endl;
        for (uint64_t j = 0; j < num_rows; ++j) {
            status = builder.Append(src, elem_size);
            if (!status.ok())
                std::cout << "Status: " << status.ToString() << " at builder.Append" << std::endl;
            src = src + elem_size; // data[j][i]
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
        char* _keys = (char *) malloc(sizeof(uint64_t*) + sizeof(uint32_t) + sizeof(uint64_t));
        char* _k = _keys;
        // Copy data
        //UUID
        uint64_t *c_uuid  = new uint64_t[2]{*storage_id, *(storage_id + 1)};
        // [0] = storage_id.time_and_version;
        // [1] = storage_id.clock_seq_and_node;
        memcpy(_k, &c_uuid, sizeof(uint64_t*)); //storage_id
        _k += sizeof(uint64_t*);
        if ((i>0) && ((i%row_elements)==0)) cluster_id++; // cluster_id = i / row_elements
        memcpy( _k, &cluster_id, sizeof(uint32_t)); //cluster_id
        _k += sizeof(uint32_t);
        memcpy(_k, &i, sizeof(uint64_t)); //col_id
    //int64_t token = murmur(storage_id, cluster_id);
    //char *host = storage->get_host_per_token(token);
    //    std::cout<< "store_numpy_into_cas_as_arrow storage_id="<<std::hex<<*storage_id<<" cluster_id="<<std::dec<<cluster_id<<", col_id="<<i<<"-->"<<token<<" @"<<host<<std::endl;

        // Allocate memory for values
        uint32_t values_size = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(char*);
        char* _values = (char*) malloc(values_size);
        char* _val = _values; // Temporal to avoid modifications on the original value
        // Copy data
        uint64_t row_id = 0;
        memcpy(_val, &row_id, sizeof(uint64_t));    //row_id
        _val += sizeof(uint64_t);
        memcpy(_val, &elem_size, sizeof(uint32_t)); //elem_size
        _val += sizeof(uint32_t);

        uint64_t arrow_size = result->size();
        char *mypayload = (char *)malloc(sizeof(uint64_t) + arrow_size);
        memcpy(mypayload, &arrow_size, sizeof(uint64_t));
        memcpy(mypayload + sizeof(uint64_t), result->data(), arrow_size);

        memcpy(_val, &mypayload, sizeof(char*));    //payload

        cache->put_crow( (void*)_keys, (void*)_values ); //Send column to cassandra
    }
}

/***
 * Write a complete numpy ndarray by columns (using arrow)
 * @param storage_id	UUID identifying the numpy ndarray
 * @param np_metas		ndarray characteristics
 * @param numpy			Memory for the whole numpy whose columns must be stored (columns consecutive in memory)
 * @param cols			Vector of columns ids to store
 */
void ArrayDataStore::store_numpy_into_cas_by_cols_as_arrow(const uint64_t *storage_id,
                                                   ArrayMetadata &metadata, void *data,
                                                   std::vector<uint32_t> &cols) const {

    throw ModuleException("NOT IMPLEMENTED YET");

    assert( metadata.dims.size() <= 2 ); // First version only supports 2 dimensions

    // Calculate row and element sizes
    uint64_t row_size   = metadata.strides[0];
    uint32_t elem_size  = metadata.elem_size;

    // Calculate number of rows and columns
    uint64_t num_columns = cols.size();
    uint64_t num_rows    = metadata.dims[0];


    // FIXME Encapsulate the following code into a function f(data, columns) -> Arrow

    uint32_t cluster_id = 0;
    uint32_t row_elements = get_row_elements(metadata);

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
        char* _keys = (char *) malloc(sizeof(uint64_t*) + sizeof(uint32_t) + sizeof(uint64_t));
        char* _k = _keys;
        // Copy data
        //UUID
        uint64_t *c_uuid  = new uint64_t[2]{*storage_id, *(storage_id + 1)};
        // [0] = storage_id.time_and_version;
        // [1] = storage_id.clock_seq_and_node;
        memcpy(_k, &c_uuid, sizeof(uint64_t*)); //storage_id
        _k += sizeof(uint64_t*);
        if ((i>0) && ((i%row_elements)==0)) cluster_id++; // cluster_id = i % row_elements
        memcpy( _k, &cluster_id, sizeof(uint32_t)); //cluster_id
        _k += sizeof(uint32_t);
        memcpy(_k, &i, sizeof(uint64_t)); //col_id
        //std::cout<< "store_numpy_into_cas_by_cols_as_arrow storage_id="<<*c_uuid<<"cluster_id="<<cluster_id<<", col_id="<<i<<std::endl;

        // Allocate memory for values
        uint32_t values_size = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(char*);
        char* _values = (char*) malloc(values_size);
        char* _val = _values; // Temporal to avoid modifications on the original value
        // Copy data
        uint64_t row_id = 0;
        memcpy(_val, &row_id, sizeof(uint64_t));    //row_id
        _val += sizeof(uint64_t);
        memcpy(_val, &elem_size, sizeof(uint32_t)); //elem_size
        _val += sizeof(uint32_t);

        uint64_t arrow_size = result->size();
        char *mypayload = (char *)malloc(sizeof(uint64_t) + arrow_size);
        memcpy(mypayload, &arrow_size, sizeof(uint64_t));
        memcpy(mypayload + sizeof(uint64_t), result->data(), arrow_size);

        memcpy(_val, &mypayload, sizeof(char*));    //payload

        //cache_arrow_write->put_crow( (void*)_keys, (void*)_values ); //Send column to cassandra
        cache->put_crow( (void*)_keys, (void*)_values ); //Send column to cassandra
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

    //std::cout<< "store_numpy_into_cas " << std::endl;
    while (!partitions_it->isDone()) {
        Partition part = partitions_it->getNextPartition();
        //std::cout<< "  cluster_id " << part.cluster_id << " block_id "<<part.block_id<< std::endl;
        store_numpy_partition_into_cas(storage_id, part);
    }
    //this->partitioner.serialize_metas();
    delete (partitions_it);

    // No need to flush the elements because the metadata are written after the data thanks to the queue
}

/* get_cluster_ids - Returns a 'list' with the cluster identifiers (it may be empty) */
std::list<int32_t> ArrayDataStore::get_cluster_ids(ArrayMetadata &metadata) const{
    std::set<int32_t> clusters = {};
    SpaceFillingCurve::PartitionGenerator* partitions_it = this->partitioner.make_partitions_generator(metadata, NULL); // For this case the 'data' is not needed because we only need the cluster_ids

    while (!partitions_it->isDone()) { clusters.insert(partitions_it->computeNextClusterId()); }
    std::list<int32_t> result = {};
    for(int32_t x : clusters) { result.push_back(x); };
    return result;
}

/* get_block_ids - Returns a 'list' with the cluster + block identifiers (it may be empty) */
std::list<std::tuple<uint64_t, uint32_t, uint32_t, std::vector<uint32_t> >> ArrayDataStore::get_block_ids(ArrayMetadata &metadata) const{
    // TODO Create a new iterator instead of using this 'data = NULL' hack
    SpaceFillingCurve::PartitionGenerator* partitions_it = this->partitioner.make_partitions_generator(metadata, NULL); // For this case the 'data' is not needed because we only need the cluster_ids

    std::list<std::tuple<uint64_t, uint32_t, uint32_t, std::vector<uint32_t>>> result = {};
    while (!partitions_it->isDone()) {
        PartitionIdxs res = partitions_it->getNextPartitionIdxs();
        std::tuple<uint64_t, uint32_t, uint32_t, std::vector<uint32_t>> ids  = std::make_tuple((int64_t)res.id, res.cluster_id, res.block_id, res.ccs);
        result.push_back(ids);
    }

    return result;
}

void ArrayDataStore::store_numpy_into_cas_by_coords(const uint64_t *storage_id, ArrayMetadata &metadata, void *data,
                                                    std::list<std::vector<uint32_t> > &coord) const {


    SpaceFillingCurve::PartitionGenerator *
    partitions_it = SpaceFillingCurve::make_partitions_generator(metadata, data, coord);

    std::set<int32_t> clusters = {};
    std::list<Partition> partitions = {};

    //std::cout<< "store_numpy_into_cas_by_coords " << std::endl;
    while (!partitions_it->isDone()) {
        auto part = partitions_it->getNextPartition();
            //std::cout<< "  cluster_id " << part.cluster_id << " block_id "<<part.block_id<< std::endl;
        partitions.push_back(part);
    }

    for (auto it = partitions.begin(); it != partitions.end(); ++it) {
    //std::cout<< "  store partition" << std::endl;
        auto part = *it;
        store_numpy_partition_into_cas(storage_id, part);
    }
    //std::cout<< "store_numpy_into_cas_by_coords done" << std::endl;
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

    SpaceFillingCurve::PartitionGenerator *
	partitions_it = this->partitioner.make_partitions_generator(metadata, nullptr);
    this->cache->flush_elements();
    bool cached = false; // Is there any cluster loaded? Needed to distinguish the case where there is no data at all
    while (!partitions_it->isDone()) {
        cluster_id = partitions_it->computeNextClusterId();
        auto ret = loaded_cluster_ids.insert((uint32_t)cluster_id);
        if (ret.first == loaded_cluster_ids.end()) {
            throw ModuleException("ERROR IN SET");
        }
        if (ret.second) { // New insert
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
        } else {
            cached = true;
        }
    }


    if (all_partitions.empty() && !cached) {
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

	SpaceFillingCurve::PartitionGenerator *
	partitions_it = SpaceFillingCurve::make_partitions_generator(metadata, nullptr, coord);
	std::list<Partition> clusters = {};
	while (!partitions_it->isDone()) {
		clusters.push_back(partitions_it->getNextPartition());
	}
	std::list<Partition>::iterator it = clusters.begin();
	for (; it != clusters.end(); ++it) {
            buffer = (char *) malloc(keys_size);
            //UUID
            c_uuid = new uint64_t[2]{*storage_id, *(storage_id + 1)};
            //[0] time_and_version;
            //[1] clock_seq_and_node;
            memcpy(buffer, &c_uuid, sizeof(uint64_t *));
            offset = sizeof(uint64_t *);
            //Cluster id
            memcpy(buffer + offset, &(*it).cluster_id, sizeof((*it).cluster_id));
            //JJblock_id
            offset += sizeof((*it).cluster_id);
            memcpy(buffer + offset, &(*it).block_id, sizeof((*it).block_id));

            //We fetch the data
            TupleRow *block_key = new TupleRow(keys_metas, keys_size, buffer);
            result = read_cache->get_crow(block_key);
            delete (block_key);
            //build cluster
            all_results.insert(all_results.end(), result.begin(), result.end());
            for (const TupleRow *row:result) { // A single row should be returned
                //block = (int32_t *) row->get_element(0);
                char **chunk = (char **) row->get_element(0);
                all_partitions.emplace_back(
                        Partition((uint32_t) (*it).cluster_id + half_int, (uint32_t) (*it).block_id + half_int, *chunk));
            }
	}

	if (all_partitions.empty()) {
		throw ModuleException("no npy found on sys");
	}
	partitions_it->merge_partitions(metadata, all_partitions, save);
	for (const TupleRow *item:all_results) delete (item);
	delete (partitions_it);

}

/* Open an 'arrow_file_name' in a local environment.
 * It always succeed or raises exception
 */
int ArrayDataStore::open_arrow_file(std::string arrow_file_name) {
    int fdIn = -1;
    long retries = 0;
    //std::cout<< "open_arrow_file "<<arrow_file_name<<std::endl;
    do {
        fdIn = open(arrow_file_name.c_str(), O_RDONLY);  //TODO handle open errors; define file name
        if (fdIn < 0) {
            int err = errno;
            char buff[4096];
            sprintf(buff, "open error %s", arrow_file_name.c_str());
            perror(buff);
            if (err == ENOENT) { //File does not exist... retry
                retries ++;
                std::cout << "open_arrow_file  retry " << retries << "/"<< MAX_RETRIES << std::endl;
            }
            if ((err != ENOENT) || (retries == MAX_RETRIES)) {
                char buff[4096];
                sprintf(buff, "open error %s", arrow_file_name.c_str());
                perror(buff);
                throw ModuleException("open error "+ arrow_file_name);
            }
        }
    } while(fdIn<0);
    return fdIn;
}

#define PORT "3490" // the port client will be connecting to 

//#define MAXDATASIZE 100 // max number of bytes we can get at once 
#define MAXDATASIZE 4096 // max number of bytes we can get at once 

// get sockaddr, IPv4 or IPv6:
void *get_in_addr(struct sockaddr *sa)
{
    if (sa->sa_family == AF_INET) {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }

    return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

/* Copy file 'dst'@'host' to 'src'
 * Uses the 'scp' function and the logged user
 */
int scp(const char *host, const char *src, const char *dst) {
/*
    //fprintf(stdout, " Remote copy %s from host %s to path %s\n", src, host, dst);
    // Get USERNAME
    char *user;
    user = getenv("LOGNAME");
    if ( user == NULL ) {
        std::cerr<<" scp: User name unavailable"<<std::endl;
        perror("scp: getenv");
        exit(1);
    }

    // Run scp user@host:src dst
#define MAX_CMD_LEN 1024
    char parsrc[MAX_CMD_LEN];
    if (snprintf(parsrc, MAX_CMD_LEN, "%s@%s:%s",user,host,src)>=MAX_CMD_LEN) {
        std::cerr<<" scp: Ooops max cmd line achieved! Exitting!!!"<<std::endl;
        exit(1);
    }
    int pidh,status;
    pidh=fork();
    switch(pidh){
    case -1: perror("scp: Error forking");
             exit(1);
    case 0: execlp("scp","scp","-q",parsrc,dst,NULL);
            perror("scp: Error mutating to scp");
            exit(1);
    default: waitpid(pidh,&status,0);
            if (WIFEXITED(status)){
                if(WEXITSTATUS(status) != 0) {
                    std::cerr<<" scp -q "<<parsrc<<" "<<dst<<": error copying file" <<std::endl;
                    return -1;
                }
            } else {
                std::cerr<<" scp -q "<<parsrc<<" "<<dst<<": failed with other error copying file" <<std::endl;
                return -2;
            }
    }
    return 0;
*/

    //JJfprintf(stdout, " Remote copy %s from host %s to path %s\n", src, host, dst);
    int sockfd, numbytes;  
    char buf[MAXDATASIZE];
    struct addrinfo hints, *servinfo, *p;
    int rv;
    char s[INET6_ADDRSTRLEN];

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(host, PORT, &hints, &servinfo)) != 0) {
        std::cerr<< "getaddrinfo: " << gai_strerror(rv) << std::endl;
        return -1;
    }


    // loop through all the results and connect to the first we can
    for(p = servinfo; p != NULL; p = p->ai_next) {
        if ((sockfd = socket(p->ai_family, p->ai_socktype,
                p->ai_protocol)) == -1) {
            perror("client: socket");
            continue;
        }

        if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sockfd);
            char b[256];
            sprintf(b, "client: connect to %s:", host);
            perror(b);
            continue;
        }

        break;
    }

    if (p == NULL) {
        std::cerr<< "client: failed to connect to "<< host << std::endl;
        return -2;
    }

    inet_ntop(p->ai_family, get_in_addr((struct sockaddr *)p->ai_addr),
        s, sizeof s);
    //JJfprintf(stdout,"client: connecting to %s\n", s);

    freeaddrinfo(servinfo); // all done with this structure

    size_t filesize = 0;
    //uint8_t* mmsrc;

    int pathsize = strlen(src); //+4 -> int size (it will contain path's size)
    //JJfprintf(stdout,"path    : %s\n", src);

    if (send(sockfd, &pathsize, sizeof(pathsize), 0) == -1) { //TODO: htons --> ntoh
        perror("send");
        exit(-1);
    }

    if (send(sockfd, src, strlen(src), 0) == -1) {
        perror("send");
        exit(-1);
    }

    //JJfprintf(stdout,"before open\n");

    const char* file = strrchr(src, '/');
    char* dst_path = (char *) malloc(strlen(dst) + 1 + strlen(file));
    //char* prova = strdup(dst);
    //JJfprintf(stdout,"file: %s\n", file);
    //fprintf(stdout,"dst_path: %s\n", dst_path);
    //printf("strncat: %s\n", strncat(prova, file, strlen(file)));

    //strncat(dst_path, file, strlen(file));
    size_t i,j;
    for (i=0; i< strlen(dst); i++)
        dst_path[i] = dst[i];
    for (j=0; j< strlen(file); j++,i++)
        dst_path[i] = file[j];
    dst_path[i]='\0';
    //JJfprintf(stdout,"dst_path after strcat: %s\n", dst_path);

    int newfile = open(dst_path, O_CREAT | O_RDWR, 0600);
    if (newfile < 0) {
        perror("client: unable to open destination file");
        std::cerr << "client: Creating file " << dst << " error: "<< strerror(errno) << std::endl;
        return(-1);
    }
    //JJfprintf(stdout,"before reciving from server\n");

    numbytes = recv(sockfd, buf, MAXDATASIZE-1, 0);
    filesize += numbytes;
    while(numbytes > 0) {
        buf[numbytes] = '\0';
        //printf("client: received '%s' from server\n",buf);
        write(newfile, buf, numbytes);
        numbytes = recv(sockfd, buf, MAXDATASIZE-1, 0);
        filesize += numbytes;
    }
    if (numbytes<0) {
        std::cerr<< "RECEIVE FAILED  Remote copy " << src <<" from host " << host << " to path " << dst << std::endl;
        return(-1);
    }

    //JJfprintf(stdout,"before mmap\n");
    //JJfflush(stdout);
    //mmsrc = (uint8_t*) mmap(NULL, filesize, PROT_READ, MAP_SHARED, newfile, 0);
    //if (mmsrc == MAP_FAILED) {
    //    perror("client: unable to mmap");
    //    exit(1);
    //}
    close(newfile);
    free(dst_path);


   return 0; 

}


/* itsme: Check if 'target' hostname corresponds to this local node */
bool itsme(const char *target) {
    struct ifaddrs *result;
    getifaddrs(&result);

    char host[256];
    struct ifaddrs *rp;
    for(rp = result; rp != NULL; rp = rp->ifa_next) {
        struct sockaddr *sa = rp->ifa_addr;
        if (sa == NULL) {
            printf("no address");
        } else {
            if (sa->sa_family== AF_INET) {
                getnameinfo(sa, sizeof(struct sockaddr_in), host, 256, NULL, 0, NI_NUMERICHOST);
                if (strcmp(host, target) == 0) return true ;
                else {
                    //std::cout<< " DEBUG Target "<<target<< " is NOT host "<<host<<std::endl;
                }
            }
        }
    }

    return false;
}

int get_remote_file(const char *host, const std::string sourcepath, const std::string filename, const std::string destination_path)
{
    int r = 0;
    //std::cerr << "destination_path: " << destination_path << std::endl;
    //std::cerr << "filename: " << filename << std::endl;

    if (access((destination_path + filename).c_str(), R_OK) != 0) { //File DOES NOT exist
        size_t path_size = strlen(destination_path.c_str());
        char aux_path[path_size+1];
        char *p;

        strcpy(aux_path, destination_path.c_str());

        // Check (and create) destination_path recursively...
        for (p = aux_path+1; *p != NULL; ++p) {
            if (*p =='/') {
                *p = '\0';
                if (mkdir(aux_path, 0770) < 0) {
                    if (errno != EEXIST) {
                        std::cerr<<" scp: mkdir initial path "<< aux_path <<" failed  at "<< std::getenv("HOSTNAME") <<std::endl;
                        perror("scp: mkdir");
                        exit(1); //FIXME
                    }
                }
                *p = '/';
            }
        }
        if (mkdir(aux_path, 0770) < 0) { //last directory in path may not end with '/', so it is treated separately
            if (errno != EEXIST) {
                std::cerr<<" scp: mkdir final path "<< aux_path <<" failed  at "<< std::getenv("HOSTNAME") <<std::endl;
                perror("scp: mkdir");
                exit(1); //FIXME
            }
        }
        r = scp(host, (sourcepath + filename).c_str(), (destination_path).c_str());
    }

    return r;
}

/* Open an 'arrow_file_name' from (a distributed environment) node corresponding to partition key (storage_id, cluster_id)
 * There are 2 cases>
 *  1) The file exists LOCALLY, or
 *  2) The file is REMOTE
 * In the second case, the file is retrieved from its owner, copied locally and opened.
 * It always succeed or raises exception
 * Args:
 *      storage_id, cluster_id : partition key
 *      arrow_file_name        : KEYSPACE/TABLE path to open (relative to arrow_path)
 */
int ArrayDataStore::find_and_open_arrow_file(const uint64_t * storage_id, const uint32_t cluster_id, const std::string arrow_file_name) {

    std::string local_path = this->arrow_path;
    if (!this->arrow_optane) { // FIXME Fix this at creation time
        local_path = local_path + "arrow/";
    }

    // Create token from storage_id+cluster_id (partition key)
    int64_t token = murmur(storage_id, cluster_id);


    // Find host corresponding to token
    char *host = storage->get_host_per_token(token);

    //std::cout<< " DEBUG find_and_open_arrow_file " << local_path + arrow_file_name<< " sid="<<std::hex<<*storage_id<<" cid="<<std::dec<<cluster_id<<" -> "<<token<< " should be on host "<<host<<std::endl;

    // Detect the location of the file
    if ( !itsme(host) ) {
        std::string remote_path;
        remote_path = local_path + "REMOTES/";


        std::string ksp;
        std::string arrow_file;
        uint32_t pos = arrow_file_name.find_last_of("/");
        ksp = arrow_file_name.substr(0, pos);
        arrow_file = arrow_file_name.substr(pos, arrow_file_name.length());

        if (get_remote_file(host, local_path + ksp, arrow_file, remote_path + ksp) < 0) {
            std::string msg = " ArrayDataStore::find_and_open_arrow_file: File "
                             + (local_path + ksp + arrow_file)
                             + " does not exist remotelly at " + host + "!! ";
            throw ModuleException(msg);
        }

        // Now it is local
        local_path = remote_path;

    } else {
        //std::cout << " DEBUG token is LOCAL " << std::endl;

        if (access((local_path + arrow_file_name).c_str(), R_OK) != 0) { //File DOES NOT exist
            std::string msg = " ArrayDataStore::find_and_open_arrow_file: File "
                             + (local_path + arrow_file_name)
                             + " does not exist locally at " + host + "!! ";
            throw ModuleException(msg);
        }
    }
    return open_arrow_file(local_path + arrow_file_name);
}

/***
 * Retrieve some Numpy columns from Cassandra in Arrow format into a numpy ndarray
 * @param storage_id of the array to retrieve
 * @param metadata ndarray characteristics
 * @param cols vector of columns identifiers to get
 * @param save numpy memory object where columns will be saved (columns consecutive in memory)
 */
void ArrayDataStore::read_numpy_from_cas_arrow(const uint64_t *storage_id, ArrayMetadata &metadata,
                                                   std::vector<uint64_t> &cols, void *save) {
    if (!arrow_enabled) {
        std::cerr<< "read_numpy_from_cas_arrow called, but HECUBA_ARROW is not enabled" << std::endl;
        return;
    }
    std::shared_ptr<const std::vector<ColumnMeta> > keys_metas = read_cache->get_metadata()->get_keys();
    uint32_t keys_size = (*--keys_metas->end()).size + (*--keys_metas->end()).position;
    std::vector<const TupleRow *> result;
    uint64_t *c_uuid = nullptr;
    char *_keys = nullptr;
    int32_t offset = 0;

    std::shared_ptr<arrow::ipc::RecordBatchFileReader> sptrFileReader;

    uint64_t row_size   = metadata.strides[1]; // Columns are stored in rows, therefore even the name, this is the number of columns
    uint32_t elem_size  = metadata.elem_size;

    int fdIn = -1;
    std::string arrow_file_name;
    std::string base_arrow_file_name;
    if (this->arrow_optane) {
        //open devdax access
        arrow_file_name = std::string(this->arrow_path + "/arrow_persistent_heap");
        fdIn = open_arrow_file(arrow_file_name.c_str()); // FIXME a distributed version is needed...
    } else {
        std::string name (this->TN);
        uint32_t pos = name.find_first_of(".",0);
        name.replace(pos, 1, "/");
        base_arrow_file_name = name;
    }

    uint32_t cluster_id = 0;
    uint32_t row_elements = get_row_elements(metadata);
    for (uint32_t it = 0; it < cols.size(); ++it) {
        _keys = (char *) malloc(keys_size);
        //UUID
        c_uuid = new uint64_t[2]{*storage_id, *(storage_id + 1)};
        //[0] time_and_version;
        //[1] clock_seq_and_node;
        memcpy(_keys, &c_uuid, sizeof(uint64_t *));
        offset = sizeof(uint64_t *);
        //cluster_id
        cluster_id = cols[it] / row_elements; // Calculate manually the cluster_id...
        memcpy( _keys + offset, &cluster_id, sizeof(uint32_t)); //cluster_id
        offset += sizeof(uint32_t);
        //col id
        memcpy(_keys + offset, &cols[it], sizeof(uint64_t));
        //int64_t token = murmur(storage_id, cluster_id);
        //char *host = storage->get_host_per_token(token);
        //std::cout<< "read_numpy_from_cas_arrow storage_id="<<std::hex<<*c_uuid<<", cluster_id="<<std::dec<<cluster_id<<", col_id="<<cols[it]<<"-->"<<token<<" @"<<host<<std::endl;

        if (!this->arrow_optane) {
            arrow_file_name = base_arrow_file_name + std::to_string(cols[it]);
            fdIn = find_and_open_arrow_file( storage_id, cluster_id, arrow_file_name );
        }

        //We fetch the data
        TupleRow *block_key = new TupleRow(keys_metas, keys_size, _keys);
        result = read_cache->get_crow(block_key);// FIXME use Yolanda's IN instead of a call to cassandra per column
        delete (block_key);

        //std::cout<< " JCOSTA read_numpy_from_cas_arrow rows="<<result.size()<<std::endl;
        for (const TupleRow *row:result) { // FIXME Theoretically, there should be a single row. ENRIC ensure that any data in the buffer table for the current {storage_id, col_id} has been transfered to the arrow table! And in this case, just peek the first row from the vector
            uint64_t *arrow_addr = (uint64_t *) row->get_element(0);
            uint32_t *arrow_size = (uint32_t *) row->get_element(1);

            // CHECK len(fd_in) == arrow_size? otherwise... wait. In case a 'scp' is in flight but still not finished

            //int filesize = lseek(fdIn, 0, SEEK_END); //TODO DEBUG only
            //if (filesize < 0) {
            //    perror("hecuba: unable to lseek to end of file");
            //    exit(1);
            //}
            //std::cout << "File size from lseek: " << filesize << std::endl;

            uint32_t filesize = 0;
            int retries = 0;
            do {
                filesize = lseek(fdIn, 0, SEEK_END);
                //std::cout << "File size from lseek: " << filesize << std::endl;
                if (filesize < 0) {
                    perror("unable to lseek to end of file");
                    throw ModuleException("lseek error " + arrow_file_name);
                } else if (filesize < *arrow_size) {
                    ++retries;
                    std::cout << "File size from lseek: " << filesize << std::endl;
                    std::cout << "File size from Cassandra: " << *arrow_size << std::endl;
                    std::cout << "coherent arrow file size  retry " << retries << "/"<< MAX_RETRIES << std::endl;
                    sleep(1);
                }
                //int start_file = lseek(fdIn, 0, SEEK_SET);
                //if (start_file < 0) {
                //    perror("unable to lseek to start of file");
                //    throw ModuleException("lseek error " + arrow_file_name);
                //}

            } while (filesize < *arrow_size and retries < MAX_RETRIES);
            if (retries == MAX_RETRIES) {
                throw ModuleException("incomplete arrow file error");
            }


            //std::cout<< "read_numpy_from_cas_arrow addr="<<*arrow_addr<<" size="<<*arrow_size<<std::endl;
            off_t page_addr;

            //read from devdax
            off_t page_offset;
            //allocates in memory [...]
            size_t total_arrow_size;
            if (this->arrow_optane) {
                off_t dd_addr = *arrow_addr+PMEM_OFFSET;
                page_addr = dd_addr & ~(sysconf(_SC_PAGE_SIZE) - 1); //I don't even remember how this works

                //read from devdax
                page_offset = dd_addr-page_addr;
                //allocates in memory [...]
                total_arrow_size = *arrow_size+page_offset;
            } else { /* !OPTANE */
                page_addr = 0;
                total_arrow_size = *arrow_size;
                page_offset = 0;

            }
            auto* src = (uint8_t*) mmap(NULL, total_arrow_size, PROT_READ, MAP_SHARED, fdIn, page_addr); //TODO handle mmap errors
            if (src == MAP_FAILED) {
                throw ModuleException("mmap error");
            }

            //read from devdax
            arrow::io::BufferReader bufferReader((const uint8_t*)&src[page_offset], *arrow_size);

            arrow::Status status;
            status = arrow::ipc::RecordBatchFileReader::Open(&bufferReader, &sptrFileReader); //TODO handle RecordBatchFileReader::Open errors
            if (not status.ok()) {
                std::cerr << " OOOOPS: "<< status.message() << std::endl;
                throw ModuleException("RecordBatchFileReader::Open error");
            }

            const std::shared_ptr<arrow::ipc::RecordBatchFileReader> localPtr = sptrFileReader;
            int num_batches = localPtr->num_record_batches();

            for (int i = 0; i < num_batches; ++i) { //for each batch inside arrow File; Theoretically, there should be one batch
                std::shared_ptr<arrow::RecordBatch> chunk;
                status = localPtr->ReadRecordBatch(i, &chunk);
                if (not status.ok()) { //TODO ReadRecordBatch
                    throw ModuleException("ReadRecordBatch error");
                }
                std::shared_ptr<arrow::Array> col = chunk->column(0); //Theoretically, there must be one column

                std::shared_ptr<arrow::BinaryArray> data = std::dynamic_pointer_cast<arrow::BinaryArray>(col);

                char* dst = (char*) save;
                dst += cols[it]*row_size;

                const uint8_t* bytes = data->value_data()->data();
                memcpy(dst, bytes, col->length()*elem_size); // Copy the whole column
            }


            // TODO ENRIC create a function to translate the arrow_addr and arrow_size to memory in base[x, col[it]]

            if (munmap(src, total_arrow_size) < 0) {
                throw ModuleException("munmap error");
            }
        }
        if (!this->arrow_optane) {
            close(fdIn);
        }
    }
    for (const TupleRow *item:result) delete (item);
    sptrFileReader.reset();
    if (this->arrow_optane) {
        close(fdIn);
    }
    close(fdIn);
}
