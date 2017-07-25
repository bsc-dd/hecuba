// Copyright (c) 2006, Applied Informatics Software Engineering GmbH. and Contributors.
// SPDX-License-Identifier:	BSL-1.0

#ifndef HFETCHCache_INCLUDED
#define HFETCHCache_INCLUDED

#include "Poco/AbstractCache.h"
#include "LRUPersistentStrategy.h"

namespace Poco {

    template<
            class TKey,
            class TValue,
            class TMutex = FastMutex,
            class TEventMutex = FastMutex
    >
    class TupleRowCache : public AbstractCache<TKey, TValue, LRUPersistentStrategy<TKey, TValue>, TMutex, TEventMutex>
        /// An LRUCache implements Least Recently Used caching. The default size for a cache is 1024 entries.
    {
    public:
        TupleRowCache(long size = 1024) :
                AbstractCache<TKey, TValue, LRUPersistentStrategy<TKey, TValue>, TMutex, TEventMutex>(
                        LRUPersistentStrategy<TKey, TValue>(size)) {
        }

        ~TupleRowCache() {
        }

    private:
        TupleRowCache(const TupleRowCache &aCache);

        TupleRowCache &operator=(const TupleRowCache &aCache);
    };


} // namespace Poco


#endif // Foundation_HFECTHCache_INCLUDED
