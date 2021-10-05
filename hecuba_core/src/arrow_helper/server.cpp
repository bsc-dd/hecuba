/*
** server.c -- a stream socket server demo
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/utsname.h>

#define PORT "3490"  // the port users will be connecting to

#define BACKLOG 10   // how many pending connections queue will hold

//#define MAXDATASIZE 100
#define MAXDATASIZE 4096

void sigchld_handler(int s)
{
    // waitpid() might overwrite errno, so we save and restore it:
    int saved_errno = errno;

    while(waitpid(-1, NULL, WNOHANG) > 0);

    errno = saved_errno;
}


// get sockaddr, IPv4 or IPv6:
void *get_in_addr(struct sockaddr *sa)
{
    if (sa->sa_family == AF_INET) {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }

    return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

int get_int_from_asciihex(char* msg) {
    char asciihex[4];
    sprintf(asciihex, "%.4s", msg);

    int x;
    sscanf(asciihex, "%x", &x);
    return x;
}

int main(int argc, char *argv[])
{
    int sockfd, new_fd;  // listen on sock_fd, new connection on new_fd
    struct addrinfo hints, *servinfo, *p;
    struct sockaddr_storage their_addr; // connector's address information
    socklen_t sin_size;
    struct sigaction sa;
    int yes=1;
    char s[INET6_ADDRSTRLEN], s2[INET6_ADDRSTRLEN];
    int rv;

    char buf[MAXDATASIZE];
    char path[MAXDATASIZE];
    int numbytes, bytesread, pathsize;

/*JJ
    //stdout redirection; argv[1] -> output file
    int stdout_fd = dup(STDOUT_FILENO);
    if (argc == 2) {
        fflush(stdout);
        int redir_fd = open(argv[1], O_CREAT | O_RDWR, 0660);
        dup2(redir_fd, STDOUT_FILENO);
        close(redir_fd);
    }
    printf("RECEIVED ARGS ARE:\n");
    for (int i = 0; i < argc; ++i) {
        printf("%s ", argv[i]);
    } printf("\n");
    fflush(stdout);
    */

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
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
            exit(1);
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
        exit(1);
    }

    if (listen(sockfd, BACKLOG) == -1) {
        perror("listen");
        exit(1);
    }

    sa.sa_handler = sigchld_handler; // reap all dead processes
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGCHLD, &sa, NULL) == -1) {
        perror("sigaction");
        exit(1);
    }

    char hostname[256];
    if (gethostname(&hostname[0], 256) < 0) {
        perror("gethostname");
        exit(1);
    }
    printf("server [%s]: waiting for connections...\n\n", hostname);

    while(1) {  // main accept() loop
        sin_size = sizeof their_addr;
        new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);
        if (new_fd == -1) {
            perror("accept");
            continue;
        }

        inet_ntop(their_addr.ss_family,
            get_in_addr((struct sockaddr *)&their_addr),
            s, sizeof s);
        //JJprintf("server: got connection from %s\n", s);
        //JJfflush(stdout);

        if (!fork()) { // this is the child process
            close(sockfd); // child doesn't need the listener
            
            //numbytes = recv(new_fd, buf, MAXDATASIZE-1, 0);
            numbytes = recv(new_fd, &pathsize, sizeof(pathsize), 0); // --> ntohs
            //printf("FINAL pathsize: %i\n", pathsize);
            bytesread = numbytes;
            char* path = (char*)malloc(pathsize+1);
            numbytes = recv(new_fd, path, pathsize, 0);
            path[pathsize] = '\0';
            //JJprintf("Path received: %s\n", path);

            //useful
            int newfile = open(path, O_RDONLY);
            if (newfile > 0) {
                int filesize = lseek(newfile, 0, SEEK_END); //TODO DEBUG only
                if (filesize < 0) {
                    perror("server: unable to lseek to end of file");
                    exit(1);
                }
                //JJprintf("Bytes to send: %i. Sending file now...\n", filesize); //TODO DEBUG only

                filesize = lseek(newfile, 0, SEEK_SET); //TODO DEBUG only
                if (filesize < 0) {
                    perror("server: unable to lseek to start of file");
                    exit(1);
                }

                int total_bytes = 0; //TODO DEBUG only
                int total_sends = 0; //TODO DEBUG only
                bytesread = read(newfile, buf, MAXDATASIZE-1);
                total_bytes += bytesread; //TODO DEBUG only
                while(bytesread>0) {
                    send(new_fd, buf, bytesread, 0);
                    ++total_sends; //TODO DEBUG only
                    bytesread = read(newfile, buf, MAXDATASIZE-1); 
                    total_bytes += bytesread; //TODO DEBUG only
                }
                //JJprintf("Sent File: %s\n", path);
                //JJprintf("Total bytes sent to %s: %i. Total sends performed: %i\n\n", s, total_bytes, total_sends); //TODO DEBUG only
            } else {
                perror("Opening file");
                printf("Unable to open %s\n",path);
            }
            free(path);
                

            //JJprintf("\n");
            close(new_fd);
            exit(0);
        }
        close(new_fd);  // parent doesn't need this
    }

    //JJfflush(stdout);
    //JJdup2(stdout_fd, STDOUT_FILENO);
    //JJclose(stdout_fd);

    return 0;
}
