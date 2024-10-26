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
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* !  _GNU_SOURCE */
#include <sched.h>
#include <sys/select.h>


#define DBG(X) do {\
		std::cerr << "CASS_MGR [" << getpid() << "] "<< X << std::endl;\
	} while (0);

#define PORT "6666"  // the port users will be connecting to

#define BACKLOG 10   // how many pending connections queue will hold

//#define MAXDATASIZE 100
#define MAXDATASIZE 4096

int cassandraPID=0;
cpu_set_t cassandraMask; /* Original cassandra mask */
cpu_set_t currentCassandraMask; /* Current cassandra mask */

/* PROTOCOL COMMANDS */
enum cmd_state {
	ADD,
	REMOVE,
	END
};

char * cmd_str[] = {
	"ADD",
	"REMOVE",
	"END"
};
/* PROTOCOL COMMANDS END */

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

// Sets 'cassandraMask' as the cassandra mask
int setCassandraAfinity(const cpu_set_t* cassandraMask) {
	// TODO setaffinity for all cassandra processes
	if (sched_setaffinity(cassandraPID, sizeof(cpu_set_t), cassandraMask) == -1) {
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


	char hostname[256];
	if (gethostname(&hostname[0], 256) < 0) {
		perror("gethostname");
		exit(1);
	}

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

					int cmd;
					int numbytes = recv(i, &cmd, sizeof(cmd), 0); // --> ntohs
   					DBG("server: received command "<<ntohl(cmd)<<" with " <<sizeof(cmd)<<" bytes using "<< numbytes << " bytes");
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
						cmd = ntohl(cmd);
						if ( cmd > END ) {
							DBG("ERROR: Unknown command received ["<< cmd << "]. Ignored.");
							continue;
						}
						DBG(" Received cmd: " << std::string(cmd_str[cmd]) << " from "<< new_fd);
						switch (cmd) {
							case ADD:
								{
									// Receive SET SIZE
									int set_size;
									int numbytes = recv(new_fd, &set_size, sizeof(set_size), 0); // --> ntohs
									set_size = ntohl(set_size);
   									DBG("server: received size "<<set_size<<"/"<<sizeof(cpu_set_t)<<" with " <<sizeof(set_size)<<" bytes using "<< numbytes << " bytes");
									// Receive SET itself
									cpu_set_t s;
									numbytes = recv(new_fd, &s, set_size, 0);
   									DBG("server: received cpuset with size " <<set_size<<" bytes using "<< numbytes << " bytes");
									addMask(&s);
									break;
								}
							case REMOVE:
								{
									// Receive SET SIZE
									int set_size;
									int numbytes = recv(new_fd, &set_size, sizeof(set_size), 0); // --> ntohs
									set_size = ntohl(set_size);
   									DBG("server: received size "<<set_size<<" with " <<sizeof(set_size)<<" bytes using "<< numbytes << " bytes");
									// Receive SET itself
									cpu_set_t s;
									numbytes = recv(new_fd, &s, set_size, 0);
   									DBG("server: received cpuset with size " <<set_size<<" bytes using "<< numbytes << " bytes");
									removeMask(&s);
									int ack='1';
									//numbytes = send(new_fd, &ack, sizeof(ack), 0);
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
