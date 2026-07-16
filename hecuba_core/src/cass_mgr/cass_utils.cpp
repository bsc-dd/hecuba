#include "cass_utils.h"
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdlib>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>

// Helper variable to print cmd value
const char * cmd_str[] = {
	"ADD",
	"REMOVE",
	"END",
	"INIT"
};

char* get_region_name(char *prefix) {
#define MAX_NAME 512
    char *SHM_NAME = (char*) malloc(MAX_NAME);

    if (SHM_NAME == NULL) {
        perror("cass_utils:get_region_name: Error allocating memory for the region name");
        return NULL;
    }

    char * newID = std::getenv("UNIQ_ID");
    if (newID == NULL) {
        fprintf(stderr, "ERROR: cass_mgr: Required UNIQ_ID variable not found. Exitting.\n");
        return NULL;
    }
    snprintf(SHM_NAME, MAX_NAME, "%s_%s", prefix, newID);
    return SHM_NAME;
}

void* map_shared_mem(char* name, unsigned int size, int perms, int is_create_needed) {
        int fd = -1;
        if (is_create_needed) {
                fd = shm_open(name, O_CREAT |O_EXCL| O_RDWR, 0666);
                if (fd<0) {
                        if (errno == EEXIST)  {
                                shm_unlink(name); // Remove shared memory object and create it again
                                fd = shm_open(name, O_CREAT |O_EXCL| O_RDWR, 0666);
                        }
                }
                if (fd <0){
                        char b[512];
                        sprintf(b, "ERROR: cass_utils: Unable to create shared memory [%s]!. Aborting.",name);
                        perror(b);
                        return NULL;
                }
                if (ftruncate(fd, size) == -1) {
                        char b[512];
                        sprintf(b, "ERROR: cass_utils: Unable to truncate shared memory [%s]!. Aborting.",name);
                        perror(b);
                        close(fd);
                        return NULL;
                }
        } else {
                fd = shm_open(name,  O_RDWR, 0);
                while((fd < 0) && (errno == ENOENT)) { // Busy wait...
                        fd = shm_open(name,  O_RDWR, 0);
                }
                //fprintf(stderr, "cass_util: Open shared memory region %s with fd = %d (2)\n", name, fd);
                if (fd < 0) {
                        char b[512];
                        sprintf(b, "ERROR: cass_util: Unable to open shared memory [%s]!. Aborting.",name);
                        perror(b);
                        return NULL;
                }
        }
        // fd is open
        void *newregion = mmap(NULL, size, perms, MAP_SHARED, fd, 0);
        if (newregion == MAP_FAILED) {
                char b[512];
                sprintf(b, "ERROR: cass_util: Unable to mmap shared memory [%s]!. Aborting.",name);
                perror(b);
                newregion = NULL;
        }
        close(fd);
        return newregion;

}
