#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <linux/sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>

#include "cassandryn.h"
char SHM_NAME[255] = SHM_NAME_PREFIX;

// Original functions (real_)
static int (*real_pthread_create)(pthread_t *, const pthread_attr_t *,
                                  void *(*)(void *), void *) = NULL;
static void (*real_pthread_exit)(void *) __attribute__((noreturn)) = NULL;

static int *shared_tids = NULL;
typedef struct {
    void *(*start_routine)(void *);
    void *arg;
    const char *func_name;
    const char *file_name;
} thread_start_info_t;
static int tid_count = 0;
static pthread_mutex_t tid_mutex = PTHREAD_MUTEX_INITIALIZER;
static FILE *log_file = NULL;


static long (*real_syscall)(long, ...) = NULL;


// Write timestamp
static void log_timestamp(char* s) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    fprintf(log_file, "[%s] %s", buffer, s);
    fflush(log_file);
}

// Write thread create
static void add_tid(pid_t tid) {
    pthread_mutex_lock(&tid_mutex);
    int i=0;
    while ((i<MAX_THREADS) && (shared_tids[(tid_count+i)%MAX_THREADS] != -1)) i++;
    if (i<MAX_THREADS){
        shared_tids[(tid_count+i) % MAX_THREADS]=tid;
        tid_count=(tid_count+i+1) % MAX_THREADS;
    } else {
        fprintf(stderr, "WARNING: More than %d threads created! Ignored. Increase MAX_THREADS variable.\n", MAX_THREADS);
    }
    pthread_mutex_unlock(&tid_mutex);
}

// Write thread exit
static void remove_tid(pid_t tid) {
    pthread_mutex_lock(&tid_mutex);
    int found = 0;
    for (int i = 0; (i < MAX_THREADS) && !found; ++i) {
        found = (shared_tids[i] == tid);
        if (found)
            shared_tids[i] = -1;
    }
    pthread_mutex_unlock(&tid_mutex);
}

/* init_shared_memory: Prepares a shared memory region at SHM_NAME with an array of MAX_THREADS ints
 *                      As MAX_THREADS may change, MAX_THREADS (an integer) is
 *                      stored as the first element, so any other user of this
 *                      region knows how much memory to allocate.
 *
 *                              0      1      2                MAX-1
 *                      +------+------+------+------+-- ... --+------+
 *                      | MAX  |      |      |      |         |      |
 *                      +------+------+------+------+-- ... --+------+
 *                             ^
 *                             |
 *                             p
 *  Returns `p`
 */
static int* init_shared_memory() {
    char b[512];

    int fd = shm_open(SHM_NAME, O_CREAT |O_EXCL| O_RDWR, 0666);
    int error = (fd < 0);
    if (error) {
        if (errno == EEXIST)  {
            shm_unlink(SHM_NAME); // Remove shared memory object and create it again
            fd = shm_open(SHM_NAME, O_CREAT |O_EXCL| O_RDWR, 0666);
            error = (fd < 0);
        }
    }
    if (error){
        sprintf(b, "ERROR: cassandryn: Unable to create shared memory [%s]!. Aborting.",SHM_NAME);
        perror(b);
        return NULL;
    }
    if (ftruncate(fd, MAX_THREADS * sizeof(int)) == -1) {
        sprintf(b, "ERROR: cassandryn: Unable to truncate shared memory [%s]!. Aborting.",SHM_NAME);
        perror(b);
        close(fd);
        return NULL;
    }
    shared_tids = mmap(NULL, MAX_THREADS * sizeof(int),
                       PROT_WRITE, MAP_SHARED, fd, 0);
    if (shared_tids == MAP_FAILED) {
        sprintf(b, "ERROR: cassandryn: Unable to mmap shared memory [%s]!. Aborting.",SHM_NAME);
        perror(b);
        return NULL;
    }
    close(fd);
    for (int i = 0; i < MAX_THREADS; ++i) {
        shared_tids[i] = -1;
    }
    return shared_tids;
}

void pthread_exit(void *retval) {
    char b[512];
    sprintf(b, "THREAD EXITED: tid=%d\n", gettid());
    log_timestamp(b);
    remove_tid(gettid());
    real_pthread_exit(retval);
}

void * start_routine_wrapper( void* wrapped )  {
    char b[512];
    thread_start_info_t* w = (thread_start_info_t*) wrapped;
    int mytid=syscall(SYS_gettid);
    sprintf(b, "THREAD CREATED: tid=%d\n", mytid);
    log_timestamp(b);
    add_tid(mytid);
    void* res = w->start_routine(w->arg);
    pthread_exit(res); //to intercept exit from threads that end without an *explicit* pthread_exit
    return (void*) 0xCACA;
}

int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *), void *arg) {
	thread_start_info_t *wrapped = malloc(sizeof(thread_start_info_t));
	wrapped->start_routine = start_routine;
	wrapped->arg = arg;
	wrapped->func_name = NULL;
	wrapped->file_name = NULL;
    int res = real_pthread_create(thread, attr, start_routine_wrapper, wrapped);
    return  res;
}
void* intercept_call(char* call) {
    void* p = dlsym(RTLD_NEXT, call);
    if (p != NULL) {
        fprintf(stderr, "%s (%p) intercepted!\n", call, p);
    } else {
        fprintf(stderr, "%s NOT intercepted!\n", call);
    }
    return p;
}

static void __attribute__((constructor)) init() {
    char *log_dirname=NULL;
    char  log_name[1024];
    char *newID=NULL;

    newID=getenv("UNIQ_ID");
    if (newID == NULL) {
        fprintf(stderr, "ERROR: cassandryn: Required UNIQ_ID variable not found. Exitting.\n");
        return;
    }
    sprintf(SHM_NAME, "%s_%s",SHM_NAME_PREFIX, newID);

    log_dirname=getenv("CASSANDRA_LOG_DIR");
    if (log_dirname == NULL) {
        log_dirname=".";
    }
    sprintf(log_name,"%s/cassandryn_libtrace.log",log_dirname);

    log_file = fopen(log_name, "a");
    if (!log_file) {
        fprintf(stderr, "Failed to open %s for writing\n",log_name);
        return;
    }

    char msg[100];
    sprintf(msg, "CASSANDRYN Started with shared region at [%s]\n", SHM_NAME);
    log_timestamp(msg);

    shared_tids = init_shared_memory();
    if (shared_tids == NULL) exit(1);

    real_pthread_create = intercept_call("pthread_create");
    real_pthread_exit = intercept_call("pthread_exit");

    fprintf(stderr, "** CASSANDRYN started correctly\n");
}
static void __attribute__((destructor)) finish() {
    if (!log_file) { fclose(log_file); }
    if (!shared_tids) {
        munmap(&shared_tids[0], (MAX_THREADS)*sizeof(int));
    }
    shm_unlink(SHM_NAME); // Remove shared memory object
    fprintf(stderr, "** CASSANDRYN stopped\n");
}
