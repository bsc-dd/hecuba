#ifndef HFETCH_BUCKET_H
#define HFETCH_BUCKET_H

template <class K,class V> class StorageDict;

template <class K, class V> class Bucket {
private:
    V value;
    K key;
    StorageDict<K,V> *sd;
public:
    Bucket() {
    }

    Bucket(void *storage, const K &key) {
        sd =(StorageDict<K,V>*)storage;
        this->key = key;

    }


    K getKey() const{
        return key;
    }

    V getValue() const{
        return value;
    }

    /* COPY CONSTRUCTORS */

    Bucket(const Bucket& b) {
        this->key=b.getKey();
        this->value=b.getValue();
        this->sd = b.sd;
    }

    Bucket(const Bucket* b) {
        this->key=b->getKey();
        this->value=b->getValue();
        this->sd = b->sd;
    }

    Bucket(Bucket& b) {
        this->key=b.getKey();
        this->value=b.getValue();
        this->sd = b.sd;
    }

    Bucket(Bucket* b) {
        this->key=b->getKey();
        this->value=b->getValue();
        this->sd = b->sd;
    }

    /* OPERATORS */

    Bucket& operator=(const V& other) // copy assignment
    {
        this->value = other;
        sd->raiseupd(this);
    }


    Bucket& operator=(V& other) // copy assignment
    {
        this->value = other;
        sd->raiseupd(this);
    }

    Bucket& operator=(const V* other) // copy assignment
    {
        this->value = *other;
        sd->raiseupd(this);
    }


    Bucket& operator=(V* other) // copy assignment
    {
        this->value = *other;
        sd->raiseupd(this);
    }


    V operator->() // TODO copy assignment ?
    {
        return this->value;

    }

    operator V() const {
        return this->value;
    }

    bool operator==(const V &value) const{
        return this->getValue()==value;
    }
};


template<class K, class V> std::ostream& operator<<(std::ostream& os, const Bucket<K,V>& p)
{
    os << p.getValue();
    return os;
}
#endif //HFETCH_BUCKET_H
