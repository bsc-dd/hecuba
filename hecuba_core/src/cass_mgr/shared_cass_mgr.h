#ifndef SHARED_CASS_MGR_H
#define SHARED_CASS_MGR_H

#include <semaphore.h>

#define MAX_THREADS 1024

#define SHM_NAME_AFFINITY_PREFIX  "/hecuba_affinity_shm"
#define SHM_NAME_SNOOPY_PREFIX    "/hecuba_cassandryn_shm"

/* PROTOCOL COMMANDS */
enum cmd_state {
	ADD,	//ADD mask
	REMOVE,	//REMOVE mask
	END,    //END cassandra manager
	INIT	//INITIALIZED STATE
};
extern const char * cmd_str[];

struct comm_cass_mgr {
        cpu_set_t mask;     // Mask to set
        enum cmd_state op_requested;   // Pending operation requested
        enum cmd_state op_finalized;   // Already processed operation
};

struct shared_cass_mgr_data {
        unsigned cass_mgr_PID; // PID of the cass manager creating this region
        unsigned last_idx;
        sem_t last_idx_sem;
        struct comm_cass_mgr affinity_ops_state[MAX_THREADS]; 
};
#endif /* SHARED_CASS_MGR_H*/
