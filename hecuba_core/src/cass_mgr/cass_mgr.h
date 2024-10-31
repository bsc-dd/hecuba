#ifndef _CASS_MGR_H_
#define _CASS_MGR_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* !  _GNU_SOURCE */
#include <sched.h>

#define PORT "6666"  // the port users will be connecting to

#define BACKLOG 10   // how many pending connections queue will hold

//#define MAXDATASIZE 100
#define MAXDATASIZE 4096

extern int cassandraPID;
extern cpu_set_t cassandraMask; /* Original cassandra mask */
extern cpu_set_t currentCassandraMask; /* Current cassandra mask */

/* PROTOCOL COMMANDS */
enum cmd_state {
	ADD,	//ADD mask
	REMOVE,	//REMOVE mask
	END	//END cassandra manager
};

extern char * cmd_str[];

struct message {
	int 		operation;
	int 		cpusetsize;
	cpu_set_t 	set;
};

/* PROTOCOL COMMANDS END */


int setCassandraAfinity(const cpu_set_t* cassandraMask);
void initCassandraAffinity(void);
#endif
