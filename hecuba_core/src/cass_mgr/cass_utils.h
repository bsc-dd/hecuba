#ifndef CASS_UTILS_H
#define CASS_UTILS_H
#ifdef __cplusplus
extern "C" {
#endif

/* Maps region named 'name' of size 'size' bytes with permisions 'perms' (PROT_READ|PROT_WRITE).
 * The region is initially created if 'is_create_needed' is enabled.
 * RETURNS: A pointer to the mapped region or NULL */
void* map_shared_mem(char* name, unsigned int size, int perms, int is_create_needed) ;

/* Returns a new allocated name in the format : prefix_UNIQID
 * REMEMBER to free this new allocated region when not needed */
char* get_region_name(char *prefix);
#ifdef __cplusplus
}
#endif

#endif /* CASS_UTILS_H */
