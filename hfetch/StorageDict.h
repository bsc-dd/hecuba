#ifndef CPP_INTERFACE_STORAGEDICT_H
#define CPP_INTERFACE_STORAGEDICT_H


#include <map>
#include <iostream>
#include "ClusterConfig.h"
#include "TableMetadata.h"
#include "TupleRow.h"
#include "TupleRowFactory.h"
#include "CacheTable.h"
#include "Bucket.h"

template<class K, class V>
class StorageDict {
public:
    typedef std::pair<K,V> map_par;


    StorageDict(std::initializer_list<std::pair<K,V>> l) {
       for (const map_par* element= l.begin(); element!=l.end(); ++element) {
           this->mymap[element->first] = new Bucket<K,V>(this,element->first);
           this->mymap[element->first] = element->second;
       };
    }

    StorageDict(ClusterConfig *config) {
        this->cluster = config;
    }

    StorageDict() {
        this->cluster = nullptr;
    }

    ~StorageDict() {
        if (H) delete(H);
        if (KeyFactory) delete(KeyFactory);
        if (ValueFactory) delete(ValueFactory);
    }


    typedef std::pair<const K, V> value_type;
    typedef typename std::map<K,Bucket<K,V> >::iterator iterator;
    typedef typename std::map<K,Bucket<K,V> >::const_iterator const_iterator;


    /* Modifiers */

    std::pair<iterator,bool> insert (const value_type& val) {
        iterator it = mymap.find(val.first);
        if (it==mymap.end()) {
            mymap[val.first]=new Bucket<K,V>(this,val.first);
            mymap[val.first]=val.second;
            return {mymap.find(val.first), false};
        }
        else {
            return {mymap.find(val.first), true};
        }

    };



    /* Operators */

    Bucket<K,V>& operator[]( const K& key ) {
        if (mymap.find(key)==mymap.end()) mymap[key] = new Bucket<K,V>(this,key);
        return mymap[key];
    };


    Bucket<K,V>& operator[]( K&& key ) {
        if (mymap.find(key)==mymap.end()) mymap[key] = new Bucket<K,V>(this,key);
        return mymap[key];
    };



    StorageDict &operator=(std::map<K, V> somemap) {
        // Dirty method
        for (typename std::map<K,V>::const_iterator it = somemap.begin(); it!=somemap.end(); ++it) {
            K k = it->first;
            V v = it->second;
            mymap[k] = new Bucket<K,V>(this,k);
            mymap[k] = v;
        }
        if (H) this->store_data();
        return *this;
    }


    void make_persistent(std::string &name) {
        if (!cluster) throw ModuleException("StorageInterface not connected to any node");
        std::vector<std::map<std::string, std::string>> keys_names = {{{"name","key"}}};;
        std::vector<std::map<std::string, std::string>> columns_names = {{{"name","value"}}};
        TableMetadata *table_meta = new TableMetadata(name.c_str(), "my_app", keys_names, columns_names, cluster->get_session());
        std::map<std::string,std::string> config = {};
        //W = new Writer(table_meta, cluster->get_session(), config);
        H = new CacheTable(table_meta, cluster->get_session(), config);
        KeyFactory = new TupleRowFactory(table_meta->get_keys());
        ValueFactory = new TupleRowFactory(table_meta->get_values());


        this->store_data();

        mymap.clear();
    }



    void raiseupd(Bucket<K,V> *b) {
        if (H) {
            K *k = new K(b->getKey());
            V *v = new V(b->getValue());
            TupleRow *key = KeyFactory->make_tuple(k);
            TupleRow *value = ValueFactory->make_tuple(v);
            H->put_crow(key, value);
        }
    }

private:

    void store_data() {
        for (const_iterator it = mymap.begin(); it!=mymap.end(); ++it){
            K *k = new K(it->first);
            V *v = new V(it->second.getValue());
            TupleRow *key = KeyFactory->make_tuple(k);
            TupleRow *value = ValueFactory->make_tuple(v);
            H->put_crow(key,value);
        }
    }

    //Writer *W = NULL;
    CacheTable *H = nullptr;
    TupleRowFactory *KeyFactory = nullptr;
    TupleRowFactory *ValueFactory = nullptr;

    std::map<K,Bucket<K,V> > mymap;
    ClusterConfig *cluster = nullptr;
};


#endif //CPP_INTERFACE_STORAGEDICT_H
