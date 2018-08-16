#ifndef LRUPersistentStrategy_INCLUDED
#define LRUPersistentStrategy_INCLUDED

#include <Poco/LRUStrategy.h>
#include "TupleRow.h"


template<class TKey, class TValue>
class LRUPersistentStrategy : public Poco::LRUStrategy<TKey, TValue> {
public:
    LRUPersistentStrategy(std::size_t size)
            : Poco::LRUStrategy<TKey, TValue>::LRUStrategy(size) {

    }

    void onReplace(const void *, std::set<TKey> &elemsToRemove) {
        std::size_t curSize = Poco::LRUStrategy<TKey, TValue>::_keyIndex.size();

        if (curSize < Poco::LRUStrategy<TKey, TValue>::_size) {
            //Size not exceeded
            return;
        }

        std::size_t diff = curSize - Poco::LRUStrategy<TKey, TValue>::_size; //num elements to be removed
        typename Poco::LRUStrategy<TKey, TValue>::Iterator it = --Poco::LRUStrategy<TKey, TValue>::_keys.end();
        typename Poco::LRUStrategy<TKey, TValue>::IndexIterator it_v;


        while ((--it != Poco::LRUStrategy<TKey, TValue>::_keys.begin()) && elemsToRemove.size() < diff) {
            if (it->use_count() == 1) {
                it_v = Poco::LRUStrategy<TKey, TValue>::_keyIndex.find(*it);

                if (it_v != Poco::LRUStrategy<TKey, TValue>::_keyIndex.end() && it_v->second->use_count() == 1) {
                    //if values use count is also unique
                    elemsToRemove.insert(*it);
                }
            }

        }
    }

};


#endif // LRUPersistentStrategy_INCLUDED

