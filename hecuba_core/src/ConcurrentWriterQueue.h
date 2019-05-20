#ifndef HECUBA_CORE_CONCURRENTWRITERQUEUE_H
#define HECUBA_CORE_CONCURRENTWRITERQUEUE_H

#include "unordered_map"
#include "TupleRow.h"
#include "tbb/concurrent_vector.h"


template<class TKey, class TValue>
class ConcurrentWriterQueue {
public:
    ConcurrentWriterQueue(uint32_t capacity) {

    }

    void insert(TKey keys, TValue values) {

    }

    //blocking _on_free_pos
    void update(TKey keys, TValue values) {

    }

    const TupleRow* read(TKey keys) {
        //find and copy value map
        // if pos in range(-) read

        /*bool found = false;
        tbb::concurrent_vector<std::pair<TKey, TValue> >::const_iterator *start = pending_writes.cbegin();
        for (;start!=pending_writes.cend() && !found; ++start) {
            if ()
        }


        const TupleRow* values = new TupleRow(it->second);*/

    }



private:
    std::atomic<uint32_t> last_written;
    std::atomic<uint32_t> last_unprocessed;

    std::unordered_map<TKey,uint32_t > value_map;

    //tbb::concurrent_vector<std::pair<TKey, TValue> > pending_writes;
};


#endif //HECUBA_CORE_CONCURRENTWRITERQUEUE_H
