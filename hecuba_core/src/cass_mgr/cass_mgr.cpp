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

// Helper variable to print cmd value
char * cmd_str[] = {
	"ADD",
	"REMOVE",
	"END"
};

char hostname[256];
int cassandraPID=0;
cpu_set_t cassandraMask; /* Original cassandra mask */
cpu_set_t currentCassandraMask; /* Current cassandra mask */

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


#define FILE_PATH_SIZE 512
#define CHILDS_TO_CHECK_SIZE 32768

	char file_path[FILE_PATH_SIZE];
	/*buffer used for various things through the program, it doesn't have a specific purpose*/
	/*array of all the processe we've searched its child and the ones we still have to
	 *   current_child_to_check will hold the position in the array of the process we are currently checking
	 *   last_child will hold the position in the array of the last process we'll have to check
	 */
	int childs_to_check[CHILDS_TO_CHECK_SIZE];

	DIR* proc;
	struct dirent* dir_entry;
	int last_child;
	int error;

	childs_to_check[0] = pid;
	last_child = 1;

	// /proc/pid/task contains the list of all THREADS created by PID
	sprintf(file_path, "/proc/%d/task", pid);
	proc = opendir(file_path);
	if (proc == NULL) {
		perror("Failed to open the dir\n");
		return -1;
	}


	/*Iterate through all the directories*/
	while ((dir_entry = readdir(proc))){
		/* child processes will have a PID greater than its parent one */
		int th = atoi(dir_entry->d_name);
		if (th != pid) {
			if (last_child == CHILDS_TO_CHECK_SIZE) {
				std::cerr << " setCassandraAffinityRecursive:: Maximum number of childs arrived... Ignoring." << std::endl;
			} else {
				childs_to_check[last_child++] = th;
			}
		}
	}
	closedir(proc);

	// TODO CHECK /proc/pid/task/pid/children file for a list of children processes (Cassandra seems to avoid creating processes) And it is NOT RELIABLE! only stopped or frozen processes!


#if 0
	std::cerr << "CASS_MGR [" << getpid() << "] ";
	for (int i = 0; i < last_child; ++i){
			std::cerr<< std::dec<<i <<" ";
	}
	std::cerr << std::endl;
#endif

	for (int i = 0; i < last_child; ++i){
		/*change the mask of all the threads*/
		error = sched_setaffinity(childs_to_check[i], sizeof(cpu_set_t), newMask);
		if (error) {
			perror("Error changing the affinity mask\n");
			return -1;
		}
	}

	return 0;
}
// Sets 'cassandraMask' as the cassandra mask
int setCassandraAfinity(const cpu_set_t* cassandraMask) {
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
	return 0;
}

// Add cores in 'newMask' to currentCassandraMask
void addMask(const cpu_set_t* newMask) {
   if (cassandraPID == 0) return ; // Affinity is disabled
   DBG(" Adding mask [" << CPUSET2INT(newMask) <<"]");
   cpu_set_t mask;
   CPU_OR(&mask, newMask, &currentCassandraMask);
   DBG(" Setting affinity [" << CPUSET2INT(&mask) <<"]");
   setCassandraAfinity(&mask);
}

// Removes cores in 'newMask' from currentCassandraMask
void removeMask(const cpu_set_t* newMask) {
   if (cassandraPID == 0) return ; // Affinity is disabled
   DBG(" Removing mask [" << CPUSET2INT(newMask) <<"]");
   // Remove cores from currentCassandraMask
   cpu_set_t mask;
   CPU_XOR(&mask, &currentCassandraMask, newMask);
   CPU_AND(&mask, &currentCassandraMask, &mask);
   DBG(" Setting affinity [" << CPUSET2INT(&mask) <<"]");
   setCassandraAfinity(&mask);
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
	DBG("server [" << hostname << "]: Managing cassandra process "<< cassandraPID );
	int new_fd;
	int finish = 0;
	socklen_t sin_size;
	struct sockaddr_storage their_addr; // connector's address information
	DBG("server ["<< hostname << "]: waiting for connections at port "<<PORT);
	while(!finish) {  // main accept() loop
		read_fds = pfds; //Copy connected fds to temporal variable as 'select' modifies resulting set
		int poll_count = select(fd_max, &read_fds, NULL, NULL, NULL);
		if (poll_count == -1) {
			perror("poll");
			exit(1);
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
									int ack='1';
									numbytes = send(new_fd, &ack, sizeof(ack), 0);
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

	for(int i = 0; i < fd_max; i++) {
		if (FD_ISSET(i, &pfds)) close(i);
	}
	return 0;
}
