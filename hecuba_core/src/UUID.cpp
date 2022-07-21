#include "UUID.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

uint64_t* UUID::generateUUID(void) {
    uint64_t *c_uuid; // UUID for the new object
    c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);

    boost::uuids::random_generator gen;
    boost::uuids::uuid u = gen();

    memcpy(c_uuid, &u, 16);

    return c_uuid;
}

uint64_t* UUID::generateUUID5(const char* name) {
    /* uses sha1 */
    uint64_t *c_uuid; // UUID for the new object
    c_uuid = (uint64_t *) malloc(sizeof(uint64_t) * 2);

    boost::uuids::string_generator gen1;
    boost::uuids::uuid dns_namespace_uuid = gen1("{6ba7b810-9dad-11d1-80b4-00c04fd430c8}");
    boost::uuids::name_generator gen(dns_namespace_uuid);

    // boost::uuids::name_generator_md5 gen(boost::uuids::ns::dns());

    boost::uuids::uuid u = gen(name);

    memcpy(c_uuid, &u, 16);

    return c_uuid;
}

std::string UUID::UUID2str(uint64_t* c_uuid) {
    /* This MUST match with the 'cass_statement_bind_uuid' result */
    char str[37] = {};
    unsigned char* uuid = reinterpret_cast<unsigned char*>(c_uuid);
    //std::cout<< "HecubaSession: uuid2str: BEGIN "<<std::hex<<c_uuid[0]<<c_uuid[1]<<std::endl;
    sprintf(str,
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        uuid[0], uuid[1], uuid[2], uuid[3],
        uuid[4], uuid[5],
        uuid[6], uuid[7],
        uuid[8], uuid[9],
        uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]
        );
    //std::cout<< "HecubaSession: uuid2str: "<<str<<std::endl;
    return std::string(str);
}
