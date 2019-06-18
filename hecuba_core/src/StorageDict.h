#ifndef CPP_INTERFACE_STORAGEDICT_H
#define CPP_INTERFACE_STORAGEDICT_H


#include <map>
#include <iostream>
#include <type_traits>

#include "ClusterConfig.h"
#include "TableMetadata.h"
#include "TupleRow.h"
#include "TupleRowFactory.h"
#include "CacheTable.h"
#include "Bucket.h"

template<class K, class V>
class StorageDict {
public:
    typedef std::pair<K, V> value_type;
    typedef std::pair<const K, V> value_type_c;
    typedef typename std::map<K, Bucket<K, V> >::iterator iterator;
    typedef typename std::map<K, Bucket<K, V> >::const_iterator const_iterator;


    StorageDict() {
        this->cluster = nullptr;
    }


    StorageDict(std::initializer_list<std::pair<K, V>> l) : StorageDict(){
        for (const value_type *element = l.begin(); element != l.end(); ++element) {
            this->mymap[element->first] = Bucket<K, V>(this, element->first, element->second);
        };
    }

    explicit StorageDict(ClusterConfig *config) {
        this->cluster = config;
    }


    ~StorageDict() {
        if (H) delete (H);
        //if (KeyFactory) delete (KeyFactory);
        //if (ValueFactory) delete (ValueFactory);
    }


    /* Modifiers */

    std::pair<iterator, bool> insert(const value_type_c &val) {
        iterator it = mymap.find(val.first);
        if (it == mymap.end()) {
            mymap[val.first] = Bucket<K, V>(this, val.first, val.second);
            return {mymap.find(val.first), false};
        } else {
            return {mymap.find(val.first), true};
        }

    };



    /* Operators */

    // Access
    Bucket<K, V> &operator[](const K &key) {
        if (mymap.find(key) == mymap.end()) mymap[key] = Bucket<K, V>(this, key);
        return mymap[key];
    };

    // Access
    Bucket<K, V> &operator[](K &&key) {
        if (mymap.find(key) == mymap.end()) mymap[key] = Bucket<K, V>(this, key);
        return mymap[key];
    };


    // Assignment
    StorageDict &operator=(std::map<K, V> somemap) {
        // Dirty method
        for (typename std::map<K, V>::const_iterator it = somemap.begin(); it != somemap.end(); ++it) {
            K key = it->first;
            V val = it->second;
            mymap[key] = Bucket<K, V>(this, key, val);
        }
        if (H) this->store_data();
        return *this;
    }


    void make_persistent(std::string &name) {
        if (!cluster) throw ModuleException("StorageInterface not connected to any node");
        //is_same<int, int>::value
        create_table(name);

        std::vector<std::map<std::string, std::string>> keys_names = {{{"name", "key"}}};;
        std::vector<std::map<std::string, std::string>> columns_names = {{{"name", "value"}}};
        TableMetadata *table_meta = new TableMetadata(name.c_str(), "test", keys_names, columns_names,
                                                      cluster->get_session());
        std::map<std::string, std::string> config = {};

        H = new CacheTable(table_meta, cluster->get_session(), config);

        this->store_data();

        mymap.clear();
    }


    void raiseupd(Bucket<K, V> *b) {
        if (H) {
            K *k = new K(b->getKey());
            V *v = new V(b->getValue());
            H->put_crow(k, v);

        }
    }

private:

    void store_data() {
        for (const_iterator it = mymap.begin(); it != mymap.end(); ++it) {
            K *k = new K(it->first);
            V *v = new V(it->second.getValue());
            H->put_crow(k, v);
        }
    }

    void create_table(std::string name) {
        std::string key = "key "+get_type<K>();
        std::string value = "value "+get_type<V>();
        if (name.find('.',0)==std::string::npos) name="my_app."+name;
        std::cout << "CREATE TABLE "+ name +" IF NOT EXISTS " +key+","+value+" PRIMARY KEY(key);" << std::endl;
    }


    template <class T> std::string get_type() {
        if (std::is_same<T, int32_t >::value) return "int";
        else if ((std::is_same<T, int64_t >::value)) return "bigint";
        else if ((std::is_same<T, double >::value)) return "double";
        else if (std::is_same<T, std::string>::value || std::is_same<T, char*>::value) return "text";
        else throw ModuleException("Data type not supported yet");
    }


    CacheTable *H = nullptr;
    TupleRowFactory *KeyFactory = nullptr;
    TupleRowFactory *ValueFactory = nullptr;

    std::map<K, Bucket<K, V> > mymap;
    ClusterConfig *cluster = nullptr;
};


#endif //CPP_INTERFACE_STORAGEDICT_H
