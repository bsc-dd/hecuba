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

#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

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

        std::vector<std::map<std::string, std::string>> read_keys_names(keys_names.begin(), (keys_names.end() - 1));
        std::vector<std::map<std::string, std::string>> read_columns_names = columns_names;
        read_columns_names.insert(read_columns_names.begin(), keys_names.back());

        table_meta = new TableMetadata(table, keyspace, read_keys_names, read_columns_names, session);
        this->read_cache = new CacheTable(table_meta, session, config);

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

/* Generate a token for composite key (storage_id, cluster_id)
 * Args:
 *      storage_id: Points to a CassUuid
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
            //Transform the fields from the cassandra 'CassUuid' type. Check also parse_uuid from HCache.cpp
            uint32_t time_low = htobe32((*storage_id) & 0xFFFFFFFF);
            uint16_t time_mid = htobe16(((*storage_id)>>32) & 0xFFFF);
            uint16_t time_hi  = htobe16(((*storage_id)>>48));
            sidBE_high        = (time_low | ((uint64_t) time_mid)<<32 | ((uint64_t)time_hi)<<48 );

            uint64_t node     = htobe64((*(storage_id+1)) & 0xFFFFFFFFFFFF);
            uint16_t clock    = htobe16((*(storage_id+1))>>48);
            sidBE_low         = node | clock; //This works because 'node' has been translated to BigEndian

            cidBE = htobe32(cluster_id);
            sidlenBE = htobe16(sidlenBE);
            cidlenBE = htobe16(cidlenBE);
        }
        char* p = mykey;
        memcpy(p, &sidlenBE, sizeof(sidlenBE)); // The 'CassUuid' type
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
    uint64_t num_columns = metadata.dims[0];
    uint64_t num_rows    = metadata.dims[1];

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
        std::cout<< "store_numpy_into_cas_by_cols_as_arrow storage_id="<<*c_uuid<<"cluster_id="<<cluster_id<<", col_id="<<i<<std::endl;

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

    if (metadata.partition_type != COLUMNAR) {
        SpaceFillingCurve::PartitionGenerator *partitions_it = this->partitioner.make_partitions_generator(metadata, data);

        while (!partitions_it->isDone()) {
            Partition part = partitions_it->getNextPartition();
            store_numpy_partition_into_cas(storage_id, part);
        }
        //this->partitioner.serialize_metas();
        delete (partitions_it);

        // No need to flush the elements because the metadata are written after the data thanks to the queue

    } else {
        store_numpy_into_cas_as_arrow(storage_id, metadata, data);
    }
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


    if (metadata.partition_type == COLUMNAR) {
        // FIXME store_numpy_into_cas_by_coords_as_arrow(storage_id, metadata, data, coord);
        //		 if we decide to store partial columns
        throw ModuleException("Unexpected case: Are you calling store_numpy_from_cas with an Arrow format?");
    }
    SpaceFillingCurve::PartitionGenerator *
    partitions_it = SpaceFillingCurve::make_partitions_generator(metadata, data, coord);

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

    if (metadata.partition_type == COLUMNAR) {
        throw ModuleException("Unexpected case: Are you calling read_numpy_from_cas to an Arrow format");
    }
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
                                                   std::list<std::vector<uint32_t> > &coord, bool direct_copy, void *save) {

	if (metadata.partition_type == COLUMNAR) {
		throw ModuleException("Unexpected case: Are you calling read_numpy_from_cas_by_coords with an Arrow format?");
	}
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
    if (!direct_copy) {
	    partitions_it->merge_partitions(metadata, all_partitions, save);
    } else {
        uint32_t wanted_block = partitions_it->getBlockID(coord.front());
        for ( Partition p : all_partitions ) {
            // A single block is supported, all the others are DISCARDED
            if (p.block_id == wanted_block) {
                char *input = (char *) p.data;
                uint64_t *retrieved_block_size = (uint64_t *) input;
                input += sizeof(uint64_t);
                memcpy(save, input, *retrieved_block_size);
                break;
            }
        }
    }
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
                sleep(1);
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


/* Copy file 'dst'@'host' to 'src'
 * Uses the 'scp' function and the logged user
 */
void scp(const char *host, const char *src, const char *dst) {
    char exec[180];

    //fprintf(stdout, " Remote copy %s from host %s to path %s\n", src, host, dst);
    // Get USERNAME
    char *user;
    user = getenv("LOGNAME");
    if ( user == NULL ) {
        std::cerr<<" scp: User name unavailable"<<std::endl;
        perror("scp: getenv");
        exit(1); //FIXME
    }

    // Run scp user@host:src dst
    sprintf(exec,"scp -q %s@%s:%s %s", user, host, src, dst);
    int res = system(exec);
    if(res!=0)
        printf("\nFile %s not copied successfully\n",src);
    //else
    //    printf("\nFile %s copied successfully\n",src);
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

    // Where I am running?
    char hostname[128];
    gethostname(hostname, sizeof(hostname));
    struct hostent *ent = gethostbyname(hostname);
    struct in_addr ip_addr = *(struct in_addr *)(ent->h_addr);
    char * whoami = inet_ntoa(ip_addr);
    //printf("Hostname: %s, was resolved to: %s\n", hostname, whoami);

    // Create token from storage_id+cluster_id (partition key)
    int64_t token = murmur(storage_id, cluster_id);

    // Find host corresponding to token
    char *host = storage->get_host_per_token(token);

    // Detect the location of the file
    if ((strcmp(host, "127.0.0.1") != 0) && (strcmp(host, whoami) != 0)) { // The file is remote, get a copy //TODO improve local management instead of the 127.0.0.1 hack
        std::string remote_path;
        remote_path = local_path + "REMOTES/";

        std::string ksp;
        uint32_t pos = arrow_file_name.find_last_of("/");
        ksp = arrow_file_name.substr(0, pos);
        // Check that the file exists to avoid copy
        if (access((remote_path + arrow_file_name).c_str(), R_OK) != 0) { //File DOES NOT exist
            int r = mkdir((remote_path + ksp).c_str(), 0770);
            if (r<0) {
                if (errno != EEXIST) {
                    std::cerr<<" scp: mkdir "<< (remote_path + ksp) <<" failed "<<std::endl;
                    perror("scp: mkdir");
                    exit(1); //FIXME
                }
            //} else {
            //    std::cout<<" scp: created directory "<<(remote_path + ksp)<<std::endl;
            }

            // Get a copy of arrow_file_name to REMOTES path
            scp(host, (local_path + arrow_file_name).c_str(), (remote_path + ksp).c_str());
        //} else {
        //    std::cout<<" scp: Already existing file "<<(remote_path + arrow_file_name)<<std::endl;
        }

        // Now it is local
        local_path = remote_path;
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

    uint64_t row_size   = metadata.strides[0]; // Columns are stored in rows, therefore even the name, this is the number of columns
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
        //std::cout<< "read_numpy_from_cas_arrow storage_id="<<std::hex<<*c_uuid<<", cluster_id="<<std::dec<<cluster_id<<", col_id="<<cols[it]<<std::endl;

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
