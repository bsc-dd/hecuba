#ifndef KVCache_INCLUDED
#define KVCache_INCLUDED

#include <iostream>
#include <unordered_map>
#include <list>

/***
 * Simple implementation of an LRU Cache.
 * @tparam TKey Key indexing, must be hashable
 * @tparam TValue Value associated to the Key
 */
template<class TKey, class TValue>
class KVCache {

public:
    using value_type = typename std::pair<TKey, TValue>;
    using value_it = typename std::list<value_type>::iterator;

    KVCache(size_t size = 1024) {
        this->max_cache_size = size;
    }

    ~KVCache() {
        this->clear();
    }

    // Getters

    size_t size() const {
        return this->cache_items_map.size();
    }
    size_t get_max_cache_size() const {
        return this->max_cache_size;
    }

    const TValue &get(const TKey &key) const {
        auto it = cache_items_map.find(key);

        if (it == cache_items_map.end()) {
            throw std::out_of_range("No such key in the cache");
        } else {
            cache_items_list.splice(cache_items_list.begin(), cache_items_list,
                                    it->second);

            return it->second->second;
        }
    }


    // Modifiers

    void add(const TKey &key, const TValue &value) {
        auto it = cache_items_map.find(key);

        // Not found
        if (it == cache_items_map.end()) {
            if (cache_items_map.size() + 1 > max_cache_size) {
                // remove the last element from cache
                auto last = cache_items_list.crbegin();

                cache_items_map.erase(last->first);
                cache_items_list.pop_back();
            }

            cache_items_list.push_front(std::make_pair(key, value));
            cache_items_map[key] = cache_items_list.begin();
        } else {
            it->second->second = value;
            // the first argument should be a constant iterator but Intel fails
            cache_items_list.splice(cache_items_list.begin(), cache_items_list, it->second);
        }
    }

    void remove(const TKey &key) {
        auto it = cache_items_map.find(key);
        if (it != cache_items_map.end()) {
            cache_items_list.erase(it->second);
            cache_items_map.erase(key);
        }
    }

    void clear() {
        cache_items_map.clear();
        cache_items_list.clear();

    }


private:
    // Prevent the copy of a Cache
    KVCache(const KVCache &aCache);

    KVCache &operator=(const KVCache &aCache);

    size_t max_cache_size;

    // List ordered by access
    mutable std::list<value_type> cache_items_list;

    // Map containing references to the list items for update / read / removal purposes
    std::unordered_map<TKey, value_it> cache_items_map;
};


#endif // KVCache_INCLUDED
