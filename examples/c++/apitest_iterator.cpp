#include <iostream>

#include <StorageDict.h>
#include <KeyClass.h>
#include <ValueClass.h>
#define SIZE 3

using IntKeyClass = KeyClass<int32_t>;

using FloatValueClass = ValueClass<float>;


class MyDictClass: public StorageDict <IntKeyClass,FloatValueClass, MyDictClass> {

};

using MultipleKeyClass = KeyClass<std::string,int32_t>;

class MultipleKeyDictClass: public StorageDict <MultipleKeyClass,FloatValueClass,MultipleKeyDictClass> {

};

using StringKeyClass = KeyClass<std::string>;

class StringKeyDictClass: public StorageDict <StringKeyClass, FloatValueClass, StringKeyDictClass> {

};

void test_really_simple(const char *name) {
    MyDictClass mydict;
    int tss[SIZE] ={42, 43, 44};
    float lats[SIZE]={0.666, 0.777, 0.888};

    mydict.make_persistent(name);

    for (int i=0; i<SIZE; i++) {
	IntKeyClass k = IntKeyClass(tss[i]);
	FloatValueClass v = FloatValueClass(lats[i]);
        mydict[k] = v; 
    }

    mydict.sync();

    int i = 0;
    bool ok=true;
    int ts;
    	IntKeyClass pk;
    // iterating on dict
    for(auto it = mydict.begin(); it != mydict.end(); it++) {
	    pk=*it;
	    ts = IntKeyClass::get<0>(pk);
        if (i>=SIZE) {
            ok=false;
            break;
        } else {
            bool found = false;
            for (int j = 0; j< SIZE && !found; j++) {
                FloatValueClass v_read;
                if (tss[j] == ts){
                    found = true;
                    // double check that the retrieved key is a working key
                    v_read=mydict[pk];
                    if (lats[j] != FloatValueClass::get<0>(v_read)) {
                        ok =false;
                        break;
                    }
                }
            }
            if (!found) {
                ok = found;
                break;
            }
        }
	i++;
    }
    if (ok) {
        std::cout<<"Test really simple on keyiterator PASSED"<<std::endl;
    } else {
        std::cout<<"Test really simple on keyiterator FAILED"<<std::endl;
    }
}

void test_multiplekey(const char *name) {
    MultipleKeyDictClass mydict;


    mydict.make_persistent(name);

    const char *s[SIZE]={"how are you",
                  "I am fine",
                  "hope you are well" };
    float lats[SIZE]={0.666, 0.777, 0.888};
    int ts[SIZE]={42,43,44};


    //setting values
    for (int i=0; i<SIZE; i++) {
	MultipleKeyClass k = MultipleKeyClass(s[i],ts[i]);
	FloatValueClass v = FloatValueClass(lats[i]);
        mydict[k] = v;
    }
    mydict.sync();
    //Iterating on dict 
    int i = 0;
    bool ok=true;
    std::string it_s;
    int it_ts;
    MultipleKeyClass pk;
    for(auto it = mydict.begin(); it != mydict.end(); it++) {
	pk=*it;
	it_s = MultipleKeyClass::get<0>(pk);
        it_ts =MultipleKeyClass::get<1>(pk); 
        if (i>=SIZE) {
            ok=false;
        } else {
            bool found = false;
            for (int j = 0; j < SIZE && !found; j++) {
                if (strcmp(s[j],it_s.c_str())) {
                    found = true;
                }
            }
            if (!found) {
                ok = found;
                break;
            } else {
                for (int j = 0; j < SIZE && !found; j++) {
                    if (ts[j]==it_ts) {
                        found = true;
                    }
                }
                if (!found) {
                    ok = found;
                    break;
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

void test_string(const char *name) {
    StringKeyDictClass mydict;

    mydict.make_persistent(name);


    const char *s[SIZE]={"how are you",
                  "I am fine",
                  "hope you are well" };

    float lats[SIZE]={0.666, 0.777, 0.888};

    for (int i=0; i<SIZE; i++) {
	StringKeyClass key = StringKeyClass(s[i]);
	FloatValueClass v = FloatValueClass(lats[i]);
        mydict[key]=v;
    }

    mydict.sync();

    //Iterate
    int i = 0;
    bool ok=true;
    std::string ts;

    StringKeyClass pk;

    for(auto it = mydict.begin(); it != mydict.end(); it++) {
	pk = *it;
	ts = StringKeyClass::get<0>(pk);
        if (i>=SIZE) {
            ok=false;
        } else {
            bool found = false;
            for (int j = 0; j < SIZE && !found; j++) {
                if (strcmp(s[j],ts.c_str())) {
                    found = true;
                }
            }
            if (!found) {
                ok = found;
                break;
            }
        }
    }
    if (ok) {
        std::cout<<"Test string key on keyterator PASSED"<<std::endl;
    } else {
        std::cout<<"Test string key on keyterator FAILED "<<std::endl;
    }
}


int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;
    std::cout<< "+ Session started"<<std::endl;

    std::cout << "Starting test 1 " <<std::endl;
    test_really_simple("mydict");

    std::cout << "Starting test 2 " <<std::endl;
    test_multiplekey("mydictmultiplekey");

    std::cout << "Starting test 3 " <<std::endl;
    test_string("mydictString");

    std::cout << "End tests " <<std::endl;
}
