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

#include "debug.h"
#include <sys/time.h>



struct timeval startTV;
struct timeval stopTV;
struct timeval diff;
struct timeval acum;
unsigned long num_changes=0;



#define PORT "6666"  // the port users will be connecting to

#define BACKLOG 10   // how many pending connections queue will hold

// Helper variable to print cmd value
const char * cmd_str[] = {
	"ADD",
	"REMOVE",
	"END"
};


#define CASSANDRA_SNOOPY_SLOT_SIZE_US 1000000		/* Slot of 1ms between updates of cassandra threads pids */
/** Buffered vector for READ and WRITE. So we can read one while the other is written */
#define CHILDS_TO_CHECK_SIZE 4096
struct buffered_vector {
	int childs[CHILDS_TO_CHECK_SIZE];
	int last_child;
};
struct buffered_vector childs_to_check[2];

struct buffered_vector* childs_to_check_read  = &childs_to_check[0];
struct buffered_vector* childs_to_check_write = &childs_to_check[1];

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
	int n = childs_to_check_read->last_child;
	int *ch = childs_to_check_read->childs;
	for (int i = 0; i < n; ++i){
		num_changes++;
		/*change the mask of all the threads*/
		error = sched_setaffinity(ch[i], sizeof(cpu_set_t), newMask);
		if (error && (errno != ESRCH)) {  // the pid vector is eventually updated and may contain pids that are no longer alive
			perror("Error changing the affinity mask\n");
			return -1;
		}
		DBG("setCassandraAffinityRecursive: pid :"<<ch[i]);
	}

	return 0;
}
// Sets 'cassandraMask' as the cassandra mask
int setCassandraAfinity(const cpu_set_t* cassandraMask) {
	gettimeofday(&startTV, NULL);

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
   DBG(" Setting affinity [" << CPUSET2INT(&mask) <<"]");
   change_mask = 1;
}

// Removes cores in 'newMask' from currentCassandraMask
void removeMask(const cpu_set_t* newMask) {
   if (cassandraPID == 0) return ; // Affinity is disabled
   DBG(" Removing mask [" << CPUSET2INT(newMask) <<"]");
   // Remove cores from currentCassandraMask
   cpu_set_t mask;
   CPU_XOR(&mask, &currentCassandraMask, newMask);
   CPU_AND(&currentCassandraMask, &currentCassandraMask, &mask);
   DBG(" Setting affinity [" << CPUSET2INT(&mask) <<"]");
   change_mask = 1;
}

// Obtain cassandra Mask
void initCassandraAffinity(void) {
	bool affinityError = (cassandraPID == 0);
	if (affinityError) {
		std::cerr << " WARNING. Cassandra Affinity is DISABLED." <<std::endl;
	} else {
		CPU_ZERO(&cassandraMask);  // Clear the CPU set
		if (sched_getaffinity(cassandraPID, sizeof(cpu_set_t), &cassandraMask) == -1) {
			perror("sched_getaffinity");
			exit( -1);
		}
		DBG(" Cassandra Affinity for pid "<<cassandraPID);
		DBG(" 	["<<CPUSET2INT(&cassandraMask)<<"]");
		memcpy(&currentCassandraMask, &cassandraMask, sizeof(cpu_set_t));
	}
	timerclear(&acum);
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

int cassandra_snoopy(void* arg) {
#define FILE_PATH_SIZE 100
	char file_path[FILE_PATH_SIZE];
	DIR* proc;

	std::cerr << " Cassandra SNOOPY started!  Monitoring ["<<cassandraPID<<"]"<<std::endl;
	// /proc/pid/task contains the list of all THREADS created by PID
	sprintf(file_path, "/proc/%d/task", cassandraPID);
	proc = opendir(file_path);
	if (proc == NULL) {
		perror("Failed to open the dir\n");
		return -1;
	}
	while( ! finish_cassandra_snoopy ) {

		childs_to_check_write->childs[0] = cassandraPID;
		childs_to_check_write->last_child = 1;

		rewinddir(proc);

		struct dirent* dir_entry;
		/*Iterate through all the directories*/
		while ((dir_entry = readdir(proc))){
			if (dir_entry->d_name[0] != '.') {
				// Check that the thread pertains to some Cassandra 'Stage'
				char tmpfile[150];
				sprintf(tmpfile, "%s/%s/comm", file_path, dir_entry->d_name);
				int fd = open(tmpfile, O_RDONLY);
				if (fd<0) {
					char msgfile[180];
					sprintf(msgfile, "opening file [%s]", tmpfile);
					perror(msgfile);
					return -1;
				}
				char b[80];
				int x = read(fd, b, sizeof(b)-1);
				b[x] = '\0';

				const int exists = strstr(b, "Stage") != NULL;
				if (exists) {
					int th = atoi(dir_entry->d_name);
					DBG(" cassandra_snoopy:: " << th );
					if (childs_to_check_write->last_child == CHILDS_TO_CHECK_SIZE) {
						std::cerr << getpid()<<" cassandra_snoopy:: Maximum number of childs arrived... Ignoring." << std::endl;
					} else {
						childs_to_check_write->childs[childs_to_check_write->last_child++] = th;
					}
				}
			}
		}
		DBG(" Cassandra SNOOPY detected "<< childs_to_check_write->last_child<< "children ");
		// Swap buffered vectors
		struct buffered_vector* tmp = childs_to_check_read;
		childs_to_check_read = childs_to_check_write;
		childs_to_check_write = tmp;
		// Wait until next slot
		usleep(CASSANDRA_SNOOPY_SLOT_SIZE_US);
	}
	closedir(proc);
	DBG(" Cassandra SNOOPY finished! ");
	return 0;
}

char stack[4096];
/* start_cassandra_snoopy: Start a thread (cassandra_snoopy) sharing cass_mgr
 * memory to get the threads that Cassandra is using, updating the
 * 'childs_to_check_read' variable.
 */
void start_cassandra_snoopy() {
	int cassandra_snooppy_tid = clone(cassandra_snoopy, &stack[4096], CLONE_VM, NULL);
	if (cassandra_snooppy_tid<0) {
		perror("Creating cassandra_snoopy");
		exit(1);
	}
}

/* cass_mgr PID
 * 	PID	Cassandra PID
 * POLL code adapted from https://beej.us/guide/bgnet/html/index-wide.html
 */
int main(int argc, char *argv[])
{
	if (gethostname(&hostname[0], 256) < 0) {
		perror("gethostname");
		exit(1);
	}

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

	start_cassandra_snoopy();




	int sockfd = get_listener_socket();
	if (sockfd <0) {
		std::cerr << " Unable to get a socket "<< std::endl;
		exit(1);
	}

	// Start off with room for 5 connections
	// (We'll realloc as necessary)
	int fd_max = 0;
	fd_set read_fds; // Modifiable set for select
	fd_set pfds; //Connected file descriptors
	FD_ZERO(&pfds);

	// Add the listener to set
	FD_SET(sockfd, &pfds);
	fd_max = sockfd + 1; // For the listener

	// CASSANDRA MGR
	// 		waits a connection from a client,
	// 		then receives a cmd and a cpuset
	// 			int cmd
	// 			int cpuset_size
	// 			cpu_set_t cpuset
	//	 	finally executes 'cmd':
	//	 	ADD   : Adds mask to current mask
	//	 	REMOVE: Removes mask from current mask
	//	 	END   : Kill Cassandra mgr
	std::cerr << " === Started cassandra manager [" << hostname << "]: Managing cassandra process ["<< cassandraPID<<"]"<<std::endl;
	int new_fd;
	int finish = 0;
	socklen_t sin_size;
	struct sockaddr_storage their_addr; // connector's address information
	struct timeval timeout = {0, 1000}; //1ms
	struct timeval restimeout = {0, 1000}; // Temporal copy for timeout (as select modifies it)
	DBG("server ["<< hostname << "]: waiting for connections at port "<<PORT);
	int pending = 0;
	while(!finish) {  // main accept() loop
		read_fds = pfds; //Copy connected fds to temporal variable as 'select' modifies resulting set
		restimeout = timeout;
		int poll_count = select(fd_max, &read_fds, NULL, NULL, &restimeout);
		if (poll_count == -1) {
			perror("poll");
			exit(1);
		}
		pending += poll_count;
		if (pending > (fd_max/2)) {
				if (change_mask) {
					DBG("server ["<<hostname<<"] changing mask ["<<CPUSET2INT(&currentCassandraMask) <<"]");
   					setCassandraAfinity(&currentCassandraMask);
					change_mask = 0;
					pending = 0;
				}
		}
		if (poll_count == 0) {//Timeout
				continue;
		}


		// Run through the existing connections looking for data to read
		for(int i = 0; i < fd_max; i++) {
			if (FD_ISSET(i, &read_fds)) {
				if (i == sockfd) {
					// If listener is ready to read, handle new connection
					sin_size = sizeof their_addr;
					new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);

					if (new_fd == -1) {
						perror("accept");
					} else {

						fcntl(new_fd, F_SETFL, O_NONBLOCK); // Set NON-Blocking
						add_to_pfds(&pfds, new_fd, &fd_max);

						char s[INET6_ADDRSTRLEN];
						inet_ntop(their_addr.ss_family,
								get_in_addr((struct sockaddr *)&their_addr),
								s, sizeof(s));
						DBG("server: got connection from "<< s << " ===> "<< new_fd);
					}
				} else {
					// If not the listener, we're just a regular client

					struct message msg;
					int numbytes = recv(i, &msg, sizeof(msg), 0); // --> ntohs
					DBG("server: received "<<numbytes<<"/"<<sizeof(msg)<<" bytes from message");
					if (numbytes <= 0) {
						// Got error or connection closed by client
						if (numbytes == 0) {
							// Connection closed
							DBG("pollserver: socket "<< i << " hung up" );
						} else {
							if ((errno == EWOULDBLOCK)||(errno == EAGAIN)) {
									continue; // This is a 'select' "Feature" it may block in the recv even it says it would not...
							}
							perror("recv");
						}

						close(i); // Close socket
						del_from_pfds(&pfds, i, &fd_max);

					} else {
						// We got some good data from a client

						new_fd = i;
						int cmd = msg.operation;
						if ( cmd > END ) {
							DBG("ERROR: Unknown command received ["<< cmd << "]. Ignored.");
							continue;
						}
						DBG(" Received cmd: " << std::string(cmd_str[cmd]) << " from "<< new_fd);
						switch (cmd) {
							case ADD:
								{
									int set_size;
									set_size = msg.cpusetsize;
									DBG("server: received size "<<set_size<<"/"<<sizeof(cpu_set_t));
									addMask(&msg.set);
									break;
								}
							case REMOVE:
								{
									int set_size;
									set_size = msg.cpusetsize;
									DBG("server: received size "<<set_size<<"/"<<sizeof(cpu_set_t));
									removeMask(&msg.set);
									//int ack='1';
									//numbytes = send(new_fd, &ack, sizeof(ack), 0); // Acknowledge is not sent to ensure asynchrony
									break;
								}
							case END:
								finish = 1;
								break;
						}
					}
				}
			}
		}
	}

	// Finish cassandra_snoopy
	finish_cassandra_snoopy = 1;

	for(int i = 0; i < fd_max; i++) {
		if (FD_ISSET(i, &pfds)) close(i);
	}

	std::cerr << " === Finished cassandra manager[" << hostname << "]: Total Time = "<< acum.tv_sec <<"s "<< acum.tv_usec<<"us to execute "<<num_changes<<" sched_setaffinity" <<std::endl;
	return 0;
}
