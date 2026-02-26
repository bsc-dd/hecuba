#ifndef __STORAGE_STREAM_H
#define __STORAGE_STREAM_H
#include "IStorage.h"

namespace Hecuba {
class StorageStream: virtual public IStorage {
public:
	StorageStream () {
		this->enableStream();
	};
};
}
#endif /* __STORAGE_STREAM_H  */
