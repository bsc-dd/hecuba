#include "IStorage.h"

class StorageStream: virtual public IStorage {
public:
	StorageStream () {
		this->enableStream();
	};
};
