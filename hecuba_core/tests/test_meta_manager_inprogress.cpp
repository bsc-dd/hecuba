#include <random>
#include <cassandra.h>
#include <stdio.h>
 #include <unistd.h>
//#include "ModelOutputInfo.h"

#include "ArrayDataStore.h"
#include "SpaceFillingCurve.h"
#include "StorageInterface.h"

#define NPY_ARRAY_C_CONTIGUOUS    0x0001
#define NPY_ARRAY_F_CONTIGUOUS    0x0002
#define NPY_ARRAY_OWNDATA         0x0004
#define NPY_ARRAY_FORCECAST       0x0010
#define NPY_ARRAY_ENSURECOPY      0x0020
#define NPY_ARRAY_ENSUREARRAY     0x0040
#define NPY_ARRAY_ELEMENTSTRIDES  0x0080
#define NPY_ARRAY_ALIGNED         0x0100
#define NPY_ARRAY_NOTSWAPPED      0x0200
#define NPY_ARRAY_WRITEABLE       0x0400
#define NPY_ARRAY_UPDATEIFCOPY    0x1000

#define NUMPY_DT_FLOAT 11
#define NUMPY_DT_DOUBLE 12

typedef std::map<std::string, std::string> config_map;


void run_query(CassSession * session, std::string query) {
    CassStatement *statement = cass_statement_new(query.c_str(), 0);

    CassFuture *result_future = cass_session_execute(const_cast<CassSession *>(session), statement);
    cass_statement_free(statement);

    CassError rc = cass_future_error_code(result_future);
    if (rc != CASS_OK) {
        printf("Query execution error: %s - %s\n", cass_error_desc(rc), query.c_str());
    }
    cass_future_free(result_future);
}



int main() {
    uint32_t node_port = 9042;
    const char *contact_names = "127.0.0.1";
    uint32_t writer_queue=3, writer_parallelism=256;

    StorageInterface * SI=nullptr;
    MetaManager * MM=nullptr;
    Writer *writer = nullptr;

    char *env_path;
    env_path = std::getenv("NODE_PORT");
    if (env_path != nullptr) node_port = (uint32_t) std::atoi(env_path);

    env_path = std::getenv("CONTACT_NAMES");
    if (env_path != nullptr) contact_names = env_path;

    env_path = std::getenv("WRITE_BUFFER_SIZE");
    if (env_path != nullptr) writer_queue = (uint32_t) std::atoi(env_path);

    env_path = std::getenv("WRITE_CALLBACKS_NUMBER");
    if (env_path != nullptr) writer_parallelism = (uint32_t) std::atoi(env_path);

    try {
        SI = new StorageInterface(node_port, contact_names);
        std::cout<<"SI made" << std::endl;
    }
    catch (std::exception &e) {
        std::cerr << "Can't configure Cassandra connection: " << contact_names << ":" << node_port << std::endl;
        std::cerr << e.what() << std::endl;
        throw e;
    }

        std::string table_name("kk");
        std::string keyspace("test");
	std::string full_name ( keyspace+ "." + table_name );
	std::string query = "CREATE KEYSPACE IF NOT EXISTS " + keyspace +
               " WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1} "
	       "AND DURABLE_WRITES=false;";
    	run_query(SI->get_session(), query);

	query = "CREATE TABLE IF NOT EXISTS " + full_name +
        	" (storage_id uuid, cluster_id int, block_id int, payload blob, "
            	"PRIMARY KEY((storage_id,cluster_id),block_id)) "
            	"WITH compaction = {'class': 'SizeTieredCompactionStrategy', 'enabled': false};";
    	run_query(SI->get_session(), query);
/*** setup config ***/
    config_map config = {{"writer_par",         std::to_string(writer_parallelism)},
                         {"writer_buffer",      std::to_string(writer_queue)},
                         {"cache_size",         std::to_string(0)},
                         {"timestamped_writes", std::to_string(false)}};



    try{
        std::vector<config_map> keys_names= {{{"name","storage_id"}}};

        std::vector<config_map> columns_names={{{"name", "base_numpy"}},
						{{"name","class_name"}},
						{{"name","name"}},
						{{"name","numpy_meta"}}};
        //std::vector<config_map> columns_names={{{"name","name"}}
        //                                      ,{{"name","numpy_meta"}}};


        //std::cerr<<"Making meta manager"<<std::endl;
        //lgarrobe
        MM =SI->make_meta_manager("istorage","hecuba", keys_names, columns_names,config);

        std::cout<<"MM made" << std::endl;
    }
    catch (std::exception &e) {
        std::cerr << "Error creating meta manager" <<std::endl;
        std::cerr << e.what();
        throw e;
    }

	uint32_t n_metrics=2,lon_counter=3,n_levels=2;
	std::vector <uint32_t> dims = {n_metrics};
	std::vector <uint32_t> strides = {(uint32_t)sizeof(double)};

	/*
	std::vector <uint32_t> dims = {n_metrics, lon_counter, n_levels};
	std::vector <uint32_t> strides = {lon_counter * n_levels * (uint32_t)sizeof(double),
		n_levels * (uint32_t)sizeof(double),
		(uint32_t)sizeof(double)};
	*/
	uint32_t flags=NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED;


	ArrayMetadata arr_metas = ArrayMetadata();
	arr_metas.dims = dims;
	arr_metas.strides = strides;
	arr_metas.elem_size=sizeof(double);
	arr_metas.flags=flags;
	arr_metas.partition_type=ZORDER_ALGORITHM;
	arr_metas.typekind='f';
	arr_metas.byteorder='=';

	std::cout<< " ArrayMetadata done" << std::endl;
	std::random_device rd;
	std::mt19937_64 gen =std::mt19937_64(rd());
	std::uniform_int_distribution <uint64_t> dis;


	uint64_t *c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);

	
	c_uuid[0] = dis(gen);
	c_uuid[1] = dis(gen);

	std::cout<< " UUID  done" << std::endl;


	MM->register_obj(c_uuid, full_name.c_str(), arr_metas);

	std::cout << "JCOSTA DESPUS DE register =========" << std::endl;
	sleep(5);

        ArrayDataStore *array_store = new ArrayDataStore(table_name.c_str(), keyspace.c_str(), SI->get_session(), config);
	std::cout << "JCOSTA DESPUS DE ARRAY STORE=========" << std::endl;
	sleep(5);
#if 0



        std::vector<double> values = {2.0, 42.0};

	array_store->store_numpy_into_cas(c_uuid, arr_metas, values.data());
	std::cout << "JCOSTA DESPUS DE store numpy" << std::endl;
	sleep(5);
#endif

}
