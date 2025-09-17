#include <iostream>

#include <StorageDict.h>
#include <StorageStream.h>
#include <KeyClass.h>
#include <ValueClass.h>
#define SIZE 3

using IntKeyClass = KeyClass<int32_t>;

using FloatValueClass = ValueClass<float>;


class MyDictClass: public StorageDict <IntKeyClass,FloatValueClass, MyDictClass> , public StorageStream{

};

using MultipleKeyClass = KeyClass<std::string,int32_t>;

class MultipleKeyDictClass: public StorageDict <MultipleKeyClass,FloatValueClass,MultipleKeyDictClass>, public StorageStream {

};

using StringKeyClass = KeyClass<std::string>;

class StringKeyDictClass: public StorageDict <StringKeyClass, FloatValueClass, StringKeyDictClass>, public StorageStream {

};

void test_really_simple(const char *name, int is_producer) {
    MyDictClass mydict;
    int tss[SIZE] ={42, 43, 44};
    float lats[SIZE]={0.666, 0.777, 0.888};

    if (is_producer) {

        mydict.make_persistent(name);

        for (int i=0; i<SIZE; i++) {
            IntKeyClass k = IntKeyClass(tss[i]);
            FloatValueClass v = FloatValueClass(lats[i]);
            mydict[k] = v;
        }

        mydict.sync();
        return;
    }

    // Instantiate object
    mydict.getByAlias(name);
    int i = 0;
    bool ok=true;
    int ts;
    IntKeyClass pk;
    FloatValueClass vl;
    // iterating on dict
    for(auto it = mydict.begin(); it != mydict.end(); it++, i++) {
        pk=it->first;
        ts = IntKeyClass::get<0>(pk);
        vl=it->second;
        float ls = FloatValueClass::get<0>(vl);
        if (i>=SIZE) {
        std::cerr << " test_really_simple: oops... too many items" << std::endl;
            ok=false;
            break;
        } else {
            bool found = false;
            for (int j = 0; j< SIZE && !found; j++) {
                FloatValueClass v_read;
                if (tss[j] == ts){
                    found = true;
                    // double check that the retrieved key is a working key
                    if (lats[j] != ls) { // Check that iterator's value also is ok
                        std::cerr << " test_really_simple: oops... obtained value ["<<ls<<"] does not correspond to assigned value ["<<lats[j]<<"]" << std::endl;
                        ok = false;
                        break;
                    }
                    v_read=mydict[pk];
                    if (lats[j] != FloatValueClass::get<0>(v_read)) {
                        std::cerr << " test_really_simple: oops... obtained key does not contain assigned value [" <<lats[j]<<"]"<< std::endl;
                        ok = false;
                        break;
                    }
                }
            }
            if (!found) {
                std::cerr << " test_really_simple: oops... obtained key ["<<ts<<"] does not exist" << std::endl;
                ok = found;
                break;
            }
        }
    }
    if (i < 3) {
        std::cerr << " test_really_simple: not enough elements read :(" << std::endl;
        ok = false;
    }
    if (ok) {
        std::cout<<"Test really simple on keyiterator PASSED"<<std::endl;
    } else {
        std::cout<<"Test really simple on keyiterator FAILED"<<std::endl;
    }
}

void test_multiplekey(const char *name, int is_producer) {
    MultipleKeyDictClass mydict;

    const char *s[SIZE]={"how are you",
        "I am fine",
        "hope you are well" };
    int ts[SIZE]={42,43,44};
    float lats[SIZE]={0.666, 0.777, 0.888};

    if (is_producer) {
        mydict.make_persistent(name);



        //setting values
        for (int i=0; i<SIZE; i++) {
            MultipleKeyClass k = MultipleKeyClass(s[i],ts[i]);
            FloatValueClass v = FloatValueClass(lats[i]);
            mydict[k] = v;
        }
        mydict.sync();
        return;
    }
    // Instantiate object
    mydict.getByAlias(name);
    //Iterating on dict
    int i = 0;
    bool ok=true;
    std::string it_s;
    int it_ts;
    MultipleKeyClass pk;
    FloatValueClass vl;
    float ls;
    for(auto it = mydict.begin(); it != mydict.end(); it++) {
        pk=it->first;
        it_s = MultipleKeyClass::get<0>(pk);
        it_ts =MultipleKeyClass::get<1>(pk);
        vl=it->second;
        ls=FloatValueClass::get<0>(vl);
        if (i>=SIZE) {
            ok=false;
        } else {
            bool found = false;
            for (int j = 0; j < SIZE && !found; j++) {
                if (strcmp(s[j],it_s.c_str())==0) {
                    found = true;
                }
            }
            if (!found) {
                std::cerr << " test_multiplekey: oops... obtained key [_"<<it_s<<"_, "<<it_ts<<"] does not exist (with value ["<<ls<<"])" << std::endl;
                ok = found;
                break;
            } else {
                found = false;
                int j;
                for (j = 0; j < SIZE && !found; j++) {
                    if (ts[j]==it_ts) {
                        found = true;
                    }
                }
                if (!found) {
                    std::cerr << " test_multiplekey: oops... obtained key ["<<it_s<<", _"<<it_ts<<"_] does not exist" << std::endl;
                    ok = found;
                    break;
                }
                // double check that the retrieved key is a working value
                FloatValueClass stored;
                stored=mydict[pk];
                if (lats[j-1] != FloatValueClass::get<0>(stored)) { // At this point 'ok' == true;
                    std::cerr << " test_multiplekey: oops... obtained key does not contain assigned value [" <<lats[j-1]<<"]"<< std::endl;
                    ok = false;
                }
                if (lats[j-1] != ls) {
                    std::cerr << " test_multiplekey: oops... obtained value ["<<ls<<"] does not correspond to assigned value ["<<lats[j-1]<<"]" << std::endl;
                    ok = false;
                }
            }
        }
    }
    if (ok) {
        std::cout<<"Test multiplekey on keyiterator PASSED"<<std::endl;
    } else {
        std::cout<<"Test multiplekey on keyiterator FAILED "<<std::endl;
    }
}

void test_string(const char *name, int is_producer) {
    StringKeyDictClass mydict;

    const char *s[SIZE]={"how are you",
        "I am fine",
        "hope you are well" };

    float lats[SIZE]={0.666, 0.777, 0.888};


    if (is_producer) {
        mydict.make_persistent(name);

        for (int i=0; i<SIZE; i++) {
            StringKeyClass key = StringKeyClass(s[i]);
            FloatValueClass v = FloatValueClass(lats[i]);
            mydict[key]=v;
        }

        mydict.sync();
        return;
    }

    // Instantiate object
    mydict.getByAlias(name);
    //Iterate
    int i = 0;
    bool ok=false;
    std::string ts;

    StringKeyClass pk;

    for(auto it = mydict.begin(); it != mydict.end(); it++) {
        pk = it->first;
        ts = StringKeyClass::get<0>(pk);
            bool found = false;
            for (int j = 0; j < SIZE && !found; j++) {
                std::cout << " test_string s["<<j<<"] = "<<s[j]<<" == "<<ts<<" = ts"<<std::endl;
                if (strcmp(s[j],ts.c_str())==0) {
                    found = true;
                    ok = true;
                }
            }
            if (!found) {
                ok = false;
                break;
            }
            i++;
    }
    if (i!=SIZE) ok = false;
    if (ok) {
        std::cout<<"Test string key on keyterator PASSED"<<std::endl;
    } else {
        std::cout<<"Test string key on keyterator FAILED "<<std::endl;
    }
}


int main(int argc, char* argv[]) {
    char buffer[128];
    int producer = 0;
    //THIS FILE CHANGES BEHAVIOUR DEPENDING ON THE NAME OF THE EXECUTABLE!! // Yolanda's eyes bleed a lot
    producer = (strcmp(argv[0], "./apitest_iterator_streaming_producer")==0) ? 1 : 0;
    if (producer == 1) {
        std::cout<< "+ PRODUCER VERSION "<<std::endl;
    } else {
        std::cout<< "+ CONSUMER VERSION "<<std::endl;
    }
    std::cout<< "+ STARTING C++ APP"<<std::endl;
    std::cout<< "+ Session started"<<std::endl;

    std::cout << "Starting test 1 " <<std::endl;
    test_really_simple("mydict", producer);

    std::cout << "Starting test 3 " <<std::endl;
    test_string("mydictString", producer);
    std::cout << "Starting test 2 " <<std::endl;
    test_multiplekey("mydictmultiplekey",producer);


    std::cout << "End tests " <<std::endl;
}
