/*
** server.c -- a stream socket server demo
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <iostream>
#include <sys/utsname.h>
#include <sys/select.h>
#include <dirent.h>

#include "cass_mgr.h"
#include "cass_utils.h"

#include "debug.h"
#include <sys/time.h>
#include <sys/time.h>
#include "HecubaExtrae.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <semaphore.h>

using namespace Hecuba;

struct timeval startTV;
struct timeval stopTV;
struct timeval diff;
struct timeval acum;
unsigned long num_changes=0;
unsigned long num_failed_changes=0;
unsigned long num_adds=0;
unsigned long num_removes=0;
char *shared_array_messages_name = NULL;

cpu_set_t cassCPU_ONE;  //HARDCODED mask with ALL cpus set
cpu_set_t cassCPU_ZERO; //HARDCODED mask without any cpu


#define PORT "6666"  // the port users will be connecting to

#define BACKLOG 10   // how many pending connections queue will hold

#if 0
CASO A (CHARM)                            CASO B (FESOM)
==============                            ==============
                       +-----+                                   +-----+
                       | CM  |                                   | CM  |
                ------ +-----+                            ------ +-----+
               /                                         /         |     \
       +-----+                                   +-----+       +-----+    +-----+
       | HS  |                                   | HS  |       | HS  |    | HS  |
       +-----+                                   +-----+       +-----+    +-----+
        |  |  \                                   |              |            \
        |  |   \                                  |              |             \
        A  B    C                                 A              B              C
        0  1    2                                 2              0              1

CM necessita shared memory region with
    LAST IDX used
    cass_mgr_PID
    lock
    per thread/process (indexed by ix (map(thread_id))):
            mask
            operation_requested(ADD, REMOVE, END)
            operation_finalized(ADD, REMOVE, END)

1st time HS calls 'addCassandraAffinity' increases last IDX in CM using a lock
        -> get lock
        -> get lastIDX
        -> inc lastIDX
        -> release lock
HS necesita map <threadID -> IDX>
HS calls addCassandraAffinity
        -> check map for threadID
        --> nonexistent == 1st call
        -> store

                                REQ         FINAL               REQ
                                -1          -1
HS.addCassandraAffinity         -1          -1      -->>        ADD
HS.addCassandraAffinity         -1          ADD     -->>  (x)   ADD
HS.addCassandraAffinity         -1          REMOVE  -->>        ADD
HS.addCassandraAffinity         ADD         -1      -->>  (x)   ADD
HS.addCassandraAffinity         ADD         ADD     -->>  (x)   ADD
HS.addCassandraAffinity         ADD         REMOVE  -->>  (x)   ADD
HS.addCassandraAffinity         REMOVE      -1      *->>        ADD      (NO SE PUEDE DAR ESTE CASO)
HS.addCassandraAffinity         REMOVE      ADD     -->>        ADD
HS.addCassandraAffinity         REMOVE      REMOVE  -->>        ADD
---
HS.delCassandraAffinity         -1          -1      -->>  (x)   REMOVE
HS.delCassandraAffinity         -1          ADD     -->>        REMOVE
HS.delCassandraAffinity         -1          REMOVE  -->>  (x)   REMOVE
HS.delCassandraAffinity         ADD         -1      -->>        REMOVE     (not enough time to process ADD+REMOVE)
HS.delCassandraAffinity         ADD         ADD     -->>        REMOVE
HS.delCassandraAffinity         ADD         REMOVE  -->>        REMOVE     (not enough time to process ADD+REMOVE)
HS.delCassandraAffinity         REMOVE      -1      -->>  (x)   REMOVE
HS.delCassandraAffinity         REMOVE      ADD     -->>        REMOVE
HS.delCassandraAffinity         REMOVE      REMOVE  -->>        REMOVE

CM al despertar

                                REQ         FINAL               FINAL
                                -1          -1          -->>    -1      (nothing done)
                                -1          ADD         -->>    ADD     (nothing done)
                                -1          REMOVE      -->>    REMOVE  (nothing done)
                                ADD         -1          -->>    ADD     (change mask)
                                ADD         ADD         -->>    ADD     (nothing done)
                                ADD         REMOVE      -->>    ADD     (change mask)
                                REMOVE      -1          -->>    -1      (nothing done) (error)
                                REMOVE      ADD         -->>    REMOVE  (change mask)
                                REMOVE      REMOVE      -->>    REMOVE  (nothing done)

#endif


int CHILDS_TO_CHECK_SIZE = -1;
int* childs_to_check_read  = NULL;
struct shared_cass_mgr_data *messages_from  = NULL;

/* Buffered vector END */
int finish_cassandra_snoopy = 0 ;

char hostname[256];
int cassandraPID=0;
cpu_set_t cassandraMask; /* Original cassandra mask */
cpu_set_t currentCassandraMask; /* Current cassandra mask */
int change_mask=0;

// get sockaddr, IPv4 or IPv6:
void *get_in_addr(struct sockaddr *sa)
{
    if (sa->sa_family == AF_INET) {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }

    return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

// Add a new file descriptor to the set
void add_to_pfds(fd_set *pfds, int newfd, int *fd_max)
{
    if (newfd >= *fd_max){
	    *fd_max = newfd +1;
    }
    FD_SET(newfd, pfds);
}

// Remove an index from the set
void del_from_pfds(fd_set *pfds, int i, int *fd_max)
{
	FD_CLR(i, pfds);
	if (i == (*fd_max - 1)) { //last one... search for a previous one
		int last = -1;
		i--; // Deleted file descriptor is no more...
		for (; i > 0; i-- ) {
			if (FD_ISSET(i, pfds)) {
				last = i;
				break; // FOUND!
			}
		}
		*fd_max = (last + 1);
	}
}

// Return a string to show a cpuset
std::string CPUSET2INT(const cpu_set_t *cpuset) {
	long n = CPU_COUNT(cpuset);
	char cpus[CPU_SETSIZE];
	long last = 0;
	for (long i = 0; (n>0) && (i < CPU_SETSIZE); i++) {
		if (CPU_ISSET(i, cpuset)) {
			cpus[i] = '1';
			n--;
			last = i;
		} else {
			cpus[i] = '0';
		}
	}
	cpus[last+1]='\0';
	return std::string(cpus);
}

int setCassandraAffinityRecursive(int pid, const cpu_set_t* newMask)
{
	int error;
	char buff[10];
	int *ch = childs_to_check_read;
    cpu_set_t tmpmask;
    CPU_XOR(&tmpmask, &cassCPU_ONE, newMask); // Negate all cpus from newMask
    if (CPU_EQUAL(&tmpmask, &cassCPU_ZERO)) {//There are NO available cpus
        DBG(" --- setCassandraAffinityRecursive: unable to change mask as all of them are used");
        return 0;
    }

	for (int i = 0; i < CHILDS_TO_CHECK_SIZE ; ++i){
		/*reset the mask of all the threads*/
        if (ch[i] != -1) {
		    num_changes++;
            error = sched_setaffinity(ch[i], sizeof(cpu_set_t), &tmpmask);
            if (error && (errno != ESRCH)) {  // the pid vector is eventually updated and may contain pids that are no longer alive
                perror("Error resetting the affinity mask\n");
                return -1;
            }
            /*change the mask of all the threads (now to the desired mask forcing a migration)*/
            error = sched_setaffinity(ch[i], sizeof(cpu_set_t), newMask);
            if (error) {
                if (errno != ESRCH) {  // the pid vector is eventually updated and may contain pids that are no longer alive
                    perror("Error changing the affinity mask\n");
                    return -1;
                } else {
                    num_failed_changes ++;
                }
            }
            DBG("setCassandraAffinityRecursive: pid :"<<ch[i]);
        }
	}
	return 0;
}

// Sets 'cassandraMask' as the cassandra mask
int setCassandraAfinity(const cpu_set_t* cassandraMask) {
	gettimeofday(&startTV, NULL);
    HecubaExtrae_event(HECUBADBG, HECUBA_SETCASSAFFINITY);

	if (setCassandraAffinityRecursive(cassandraPID, cassandraMask) < 0) {
		char tmp[100];
		sprintf(tmp, "HecubaSession::setCassandraAfinity CPU_SETSIZE=%d", CPU_SETSIZE);
		std::string msg = tmp;
		if (errno == EINVAL) {
			DBG("SCHED_SET EINVAL");
			msg += " request mask=[";
			msg += CPUSET2INT(cassandraMask) + "] for pid ";
			char pid[100];
			sprintf(pid, "%d", cassandraPID);
			msg += pid;
			msg += " but current Mask is [";
			msg += CPUSET2INT(&currentCassandraMask) + "]";
		}
		perror(msg.c_str());
		return -1;
	}
	// cassandraMask = newMask;
	memcpy(&currentCassandraMask, cassandraMask, sizeof(cpu_set_t));
    HecubaExtrae_event(HECUBADBG, HECUBA_END);
	gettimeofday(&stopTV, NULL);
	timersub(&stopTV, &startTV, &diff);
	timeradd(&diff, &acum, &acum);
	return 0;
}

// Add cores in 'newMask' to currentCassandraMask
void addMask(const cpu_set_t* newMask) {
   if (cassandraPID == 0) return ; // Affinity is disabled
   DBG(" Adding mask [" << CPUSET2INT(newMask) <<"]");
   CPU_OR(&currentCassandraMask, newMask, &currentCassandraMask);
   DBG(" Setting affinity [" << CPUSET2INT(&currentCassandraMask) <<"]");
   change_mask = 1;
   num_adds++;
}

// Removes cores in 'newMask' from currentCassandraMask
void removeMask(const cpu_set_t* newMask) {
   if (cassandraPID == 0) return ; // Affinity is disabled
   DBG(" Removing mask [" << CPUSET2INT(newMask) <<"]");
   // Remove cores from currentCassandraMask
   cpu_set_t mask;
   CPU_XOR(&mask, &currentCassandraMask, newMask);
   CPU_AND(&currentCassandraMask, &currentCassandraMask, &mask);
   DBG(" Setting affinity [" << CPUSET2INT(&currentCassandraMask) <<"]");
   change_mask = 1;
   num_removes++;
}

// Obtain cassandra Mask
void initCassandraAffinity(void) {
	bool affinityError = (cassandraPID == 0);
	if (affinityError) {
		std::cerr << " WARNING. Cassandra Affinity is DISABLED." <<std::endl;
	} else {
		CPU_ZERO(&cassandraMask);  // Clear the CPU set
		if (sched_getaffinity(cassandraPID, sizeof(cpu_set_t), &cassandraMask) == -1) {
            char b[512];
            sprintf(b, "cass_mgr: sched_getaffinity failed for PID [%d]", cassandraPID);
			perror(b);
			exit( -1);
		}
		DBG(" Cassandra Affinity for pid "<<cassandraPID);
		DBG(" 	["<<CPUSET2INT(&cassandraMask)<<"]");
		memcpy(&currentCassandraMask, &cassandraMask, sizeof(cpu_set_t));
	}
	timerclear(&acum);
}

/* map_cassandra_snoopy: returns array [0..CHILDS_TO_CHECK_SIZE] integers with
 * cassandra threads' PID. The array is managed by shared memory region named
 * SHM_NAME, which is updated by *cassandryn*. */
int* map_cassandra_snoopy() {
    char* name = get_region_name(SHM_NAME_SNOOPY_PREFIX);
    if (name == NULL) {
		std::cerr << " ERROR:cass_mgr: Unable to get region name ["<<SHM_NAME_SNOOPY_PREFIX<<"]" <<std::endl;
        return NULL;
    }
    CHILDS_TO_CHECK_SIZE = MAX_THREADS;
    int *tids = (int *)map_shared_mem(name,  MAX_THREADS*sizeof(int), PROT_READ, 0);
    free(name);
    return tids;
}
void unmap_cassandra_snoopy(void* m) {
    if (!m) {
        munmap(m, MAX_THREADS*sizeof(int));
    }
}

struct shared_cass_mgr_data* map_array_messages() {
    shared_array_messages_name = get_region_name(SHM_NAME_AFFINITY_PREFIX);
    if (shared_array_messages_name == NULL) {
		std::cerr << " ERROR:cass_mgr: Unable to get region name ["<<SHM_NAME_AFFINITY_PREFIX<<"]" <<std::endl;
        return NULL;
    }
    struct shared_cass_mgr_data* region = (struct shared_cass_mgr_data*) map_shared_mem(shared_array_messages_name, sizeof(struct shared_cass_mgr_data), PROT_READ|PROT_WRITE, 1);
    return region;
}
void unmap_array_messages(void* m) {
    if (!m) {
        munmap(m, sizeof(struct shared_cass_mgr_data));
        shm_unlink(shared_array_messages_name);
        free(shared_array_messages_name);
    }
}

// Obtain a listening socket
int get_listener_socket(void) {
	int sockfd;  // listen on sock_fd
	struct addrinfo hints;
	struct addrinfo *servinfo;
	struct addrinfo *p;
	struct sigaction sa;
	int yes=1;
	int rv;
	memset(&hints, 0, sizeof hints);
	//hints.ai_family = AF_UNSPEC;
	hints.ai_family = AF_INET; //IPv4
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE; // use my IP

	if ((rv = getaddrinfo(NULL, PORT, &hints, &servinfo)) != 0) {
		fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
		return 1;
	}

	// loop through all the results and bind to the first we can
	for(p = servinfo; p != NULL; p = p->ai_next) {
		if ((sockfd = socket(p->ai_family, p->ai_socktype,
						p->ai_protocol)) == -1) {
			perror("server: socket");
			continue;
		}

		if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes,
					sizeof(int)) == -1) {
			perror("setsockopt");
			return(-1);
		}

		if (bind(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
			close(sockfd);
			perror("server: bind");
			continue;
		}


		break;
	}

	freeaddrinfo(servinfo); // all done with this structure

	if (p == NULL)  {
		fprintf(stderr, "server: failed to bind\n");
		return(-1);
	}

	if (listen(sockfd, BACKLOG) == -1) {
		perror("listen");
		return(-1);
	}
	return sockfd;

}

void initializeHardcodedMasks(void) {
	long n = CPU_COUNT(&cassCPU_ONE);
    for (int i=0; i<n; i++) {
        CPU_SET(i, &cassCPU_ONE);
    }
    CPU_ZERO(&cassCPU_ZERO);
}

void sigusr1_h (int s) {
        // Do nothing, signal only used to unblock thread
}

/* cass_mgr PID
 * 	PID	Cassandra PID
 * POLL code adapted from https://beej.us/guide/bgnet/html/index-wide.html
 */


int main(int argc, char *argv[])
{
    write(2, "CASS_MGR STARTED ========\n", 25);
	if (gethostname(&hostname[0], 256) < 0) {
		perror("gethostname");
		exit(1);
	}
    sigset_t m;
    struct sigaction sa;
    int n_ends = 0; // increments for each END operation read
    int finish = 0; // turns to true when n_ends == last_idx
    sigfillset(&m);
    sigprocmask(SIG_BLOCK, &m ,NULL);
    sigdelset(&m,SIGUSR1); // clients (HecubaSession) will send SIGUSR1 each time a new request is pending
    sa.sa_handler = sigusr1_h;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR1, &sa, NULL);

    initializeHardcodedMasks();

	DBG("=== Starting Cassandra Manager:");
	if (argc == 1) {
		DBG(" Cassandra PID missing");
		DBG(" Syntax: "<< argv[0]<< " PID ");
		return -1;
	}

	cassandraPID = atoi(argv[1]);

	DBG("RECEIVED ARGS ARE:");
	for (int i = 0; i < argc; ++i) {
		DBG( argv[i] );
	}


	initCassandraAffinity();

    childs_to_check_read = map_cassandra_snoopy();
    if (childs_to_check_read == NULL) {
        std::cerr << " ERROR: cass_mgr: Unable to map cassandra snoopy"<< std::endl;
        exit(1);
    }

    messages_from = map_array_messages();
    if (messages_from == NULL) {
        std::cerr << " ERROR: cass_mgr: Unable to map buffer for messages from threads"<< std::endl;
        exit(1);
    }



    // Initialize structure
    messages_from->cass_mgr_PID = getpid();
    messages_from->last_idx = 0;
    sem_init(&messages_from->last_idx_sem, 1, 1); // Mutex to access `last_idx`
    for (int i = 0; i< MAX_THREADS; i++) {
        messages_from->affinity_ops_state[i].op_requested = INIT;
        messages_from->affinity_ops_state[i].op_finalized = INIT;
    }

    while (!finish) {
        //std::cerr<< "CASS_MGR: Waiting for a new message"<<std::endl;
        sigsuspend(&m); // Wait a SIGUSR1 from HecubaSession
        change_mask = 0;
        for (int i = 0; i < messages_from->last_idx; i++){ // nos dejamos alguno por no recorrer siempre todos?
            enum cmd_state *op_req = &messages_from->affinity_ops_state[i].op_requested;
            enum cmd_state *op_fin = &messages_from->affinity_ops_state[i].op_finalized;
            cpu_set_t* mask = &messages_from->affinity_ops_state[i].mask;

            if ((*op_req != INIT) && (*op_fin != END)) { // there is a request
                switch (*op_req) {
                    case ADD:      if (*op_fin != ADD) {
                                       *op_fin =  ADD;
                                       addMask(mask);
                                       change_mask=1;
                                   }
                                   break;
                    case REMOVE:   if (*op_fin == ADD) {
                                       removeMask(mask);
                                       *op_fin= REMOVE;
                                       change_mask=1;
                                   }
                                   break;
                    case END:      *op_fin=END;
                                   n_ends++;
                }
            }

        }

        if (change_mask) {
            DBG("server ["<<hostname<<"] changing mask ["<<CPUSET2INT(&currentCassandraMask) <<"]");
            setCassandraAfinity(&currentCassandraMask);
        }
        finish = ((n_ends > 0) && (n_ends == messages_from->last_idx)); // coger lock sobre last_idx para compararlo? 

    }
    unmap_cassandra_snoopy((void *) childs_to_check_read);
    sem_destroy(&messages_from->last_idx_sem);
    unmap_array_messages((void *) messages_from);
std::cerr << " === Finished cassandra manager[" << hostname << "]: Total Time = "<< acum.tv_sec <<"s Received ADDs="<<num_adds<<" Received DELs="<<num_removes<<" Used "<< acum.tv_usec<<"us to execute "<<(num_changes-num_failed_changes)<<"/"<<num_changes<<" sched_setaffinity" <<std::endl;


	return 0;
}
