#ifndef __HECUBA_EXTRAE_H
#define __HECUBA_EXTRAE_H

// HECUBAEV: Type for selecting HECUBA API Events
#define HECUBAEV    42000

// HECUBA Values : IStorage / StorageDict / StorageNumpy / StorageObject
#define HECUBA_IS           0x1000
#define HECUBA_SD           0x2000
#define HECUBA_SN           0x3000
#define HECUBA_SO           0x4000
#define HECUBA_SD_KEY       0x5000
#define HECUBA_SD_VALUE     0x6000
#define HECUBA_ATTRCLASS    0x7000
#define HECUBA_SO_ATTR      0x8000

// HECUBA Values operation: (to be added to HecubaValues for example HECUBA_SD|HECUBA_END)
#define HECUBA_END              0x0
#define HECUBA_INSTANTIATION    0x1
#define HECUBA_DESTROY          0x2
#define HECUBA_READ             0x3
#define HECUBA_WRITE            0x4
#define HECUBA_SYNC             0x5
#define HECUBA_MK_PERSISTENT    0x6
#define HECUBA_ASSIGNMENT       0x7
#define HECUBA_SELECTOR         0x8
#define HECUBA_GET_BY_ALIAS     0x9

// HECUBACASS: Type for when we access cassandra
#define HECUBACASS    43000

#define HBCASS_END              0x0
#define HBCASS_READ             0x1
#define HBCASS_WRITE            0x2
#define HBCASS_DELETE           0x3
#define HBCASS_CREATE           0x4
#define HBCASS_SYNCWRITE        0x5
#define HBCASS_CONNECT          0x6
#define HBCASS_META             0x7
#define HBCASS_PREPARES         0x8

// HECUBADBG: Type to trace specific hecuba events
#define HECUBADBG    44000

#define HECUBA_TUPLEROWFACTORY  0x1
#define HECUBA_KVCACHE          0x2
#define HECUBA_REGISTER         0x3
#define HECUBA_SESSION          0x4

#ifdef EXTRAE
#include <extrae.h>

inline void HecubaExtrae_init() {
    Extrae_init();
}
inline void HecubaExtrae_event (extrae_type_t type, extrae_value_t value){
    Extrae_event(type, value);
}
inline void HecubaExtrae_comm(extrae_user_communication_types_t type, extrae_comm_id_t id) {
    struct extrae_UserCommunication write_async;
	struct extrae_CombinedEvents events;

    Extrae_init_UserCommunication(&write_async);
    write_async.type = type;
    write_async.tag = 0;
    write_async.size = 0;
    write_async.partner = 0;
    write_async.id = id;

	Extrae_init_CombinedEvents (&events);
	events.nCommunications = 1;
	events.Communications = &write_async;

	Extrae_emit_CombinedEvents (&events);
}

#else /* !EXTRAE */

#define HecubaExtrae_init()
#define HecubaExtrae_event(...) printf("Disabled Extrae_event\n");
#define HecubaExtrae_comm(...) printf("Disabled Extrae_comm\n");

#endif /* !EXTRAE */

#endif /* __HECUBA_EXTRAE_H */
