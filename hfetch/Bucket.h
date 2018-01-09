#ifndef HFETCH_BUCKET_H
#define HFETCH_BUCKET_H

template<class K, class V>
class StorageDict;

template<class K, class V>
class Bucket {
private:
    V value;
    K key;
    StorageDict<K, V> *sd = nullptr;

public:

    /* CONSTRUCTORS */

    Bucket() = default;

    Bucket(StorageDict<K, V> *storage, const K &key) {
        sd = storage;
        this->key = key;

    }

    Bucket(StorageDict<K, V> *storage, const K &key, const V &value) {
        sd = storage;
        this->key = key;
        this->value = value;

    }

    K getKey() const {
        return key;
    }

    V getValue() const {
        return value;
    }

    /* COPY CONSTRUCTORS */

    Bucket(const Bucket &b) {
        this->key = b.getKey();
        this->value = b.getValue();
        this->sd = b.sd;
    }


    Bucket(Bucket &b) {
        this->key = b.getKey();
        this->value = b.getValue();
        this->sd = b.sd;
    }

    /* OPERATORS */

    // copy assignment
    Bucket &operator=(const V &other) {
        this->value = other;
        if (sd) sd->raiseupd(this);
        return *this;
    }

    // copy assignment
    Bucket &operator=(V &other) {
        this->value = other;
        if (sd) sd->raiseupd(this);
        return *this;
    }

    // copy assignment
    Bucket &operator=(const V *other) {
        this->value = *other;
        if (sd) sd->raiseupd(this);
        return *this;
    }

    // copy assignment
    Bucket &operator=(V *other) {
        this->value = *other;
        if (sd) sd->raiseupd(this);
        return *this;
    }

    // data access pointer like
    V operator->() {
        return this->value;
    }

    // data access
    operator V() const {
        return this->value;
    }

    // comparator
    bool operator==(const V &value) const {
        return this->getValue() == value;
    }

    // comparator
    bool operator<(const V &value) const {
        return this->getValue() < value;
    }

    // comparator
    bool operator>(const V &value) const {
        return value < *this;
    }

    // comparator
    bool operator<=(const V &value) const {
        return !(value < *this);
    }

    // comparator
    bool operator>=(const V &value) const {
        return !(*this < value);
    }


};


template<class K, class V>
std::ostream &operator<<(std::ostream &os, const Bucket<K, V> &p) {
    os << p.getValue();
    return os;
}

#endif //HFETCH_BUCKET_H
