#ifndef HFETCHCache_INCLUDED
#define HFETCHCache_INCLUDED

#include "Poco/AbstractCache.h"
#include "LRUPersistentStrategy.h"

template<
        class TKey,
        class TValue,
        class TMutex = Poco::FastMutex,
        class TEventMutex = Poco::FastMutex
>
class TupleRowCache : public Poco::AbstractCache<TKey, TValue, LRUPersistentStrategy<TKey, TValue>,
        TMutex, TEventMutex> {
public:
    TupleRowCache(long size = 1024) : Poco::AbstractCache<TKey, TValue, LRUPersistentStrategy<TKey, TValue>,
            TMutex, TEventMutex>(LRUPersistentStrategy<TKey, TValue>(size)) {
    }

    ~TupleRowCache() {
    }

private:
    TupleRowCache(const TupleRowCache &aCache);

    TupleRowCache &operator=(const TupleRowCache &aCache);
};


#endif // Foundation_HFECTHCache_INCLUDED
