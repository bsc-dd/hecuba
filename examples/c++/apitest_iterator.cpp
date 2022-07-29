#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

#define SIZE 3

void test_really_simple(HecubaSession &s,const char *name) {
    IStorage* mydict = s.createObject("reallysimpledict", "mydict");
    int tss[SIZE] ={42, 43, 44};
    float lats[SIZE]={0.666, 0.777, 0.888};

    for (int i=0; i<SIZE; i++) {
        mydict->setItem(&tss[i],&lats[i]);
    }

    mydict->sync();
    int i = 0;
    bool ok=true;
    int ts;
    for(auto it = mydict->begin(); it != mydict->end(); it++) {
        ts = (int64_t)(*it);
        if (i>=SIZE) {
            ok=false;
            break;
        } else {
            bool found = false;
            for (int j = 0; j< SIZE && !found; j++) {
                if (tss[j] == ts){
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
        std::cout<<"Test really simple on keyiterator PASSED"<<std::endl;
    } else {
        std::cout<<"Test really simple on keyiterator FAILED"<<std::endl;
    }
}

void test_multiplekey(HecubaSession& mys, const char *name) {
    IStorage* mydict = mys.createObject("dictMultipleKey", name);

    //Create key
    const char *s[SIZE]={"how are you",
                  "I am fine",
                  "hope you are well" };
    float lats[SIZE]={0.666, 0.777, 0.888};
    int ts[SIZE]={42,43,44};

    for (int i=0; i<SIZE; i++) {
        char * mystring = (char *)malloc(strlen(s[i])+1);
        char * key = (char *)malloc(sizeof(char *) + sizeof(int));
        memcpy(mystring, s[i], strlen(s[i])+1);
        memcpy (key, &mystring, sizeof(char *));
        memcpy (key+sizeof(char *), &ts[i], sizeof(int));
        mydict->setItem(key,&lats[i]);
    }
    mydict->sync();

    //Iterate
    int i = 0;
    bool ok=true;
    char *it_s;
    int it_ts;
    char *buffer;
    for(auto it = mydict->begin(); it != mydict->end(); it++) {
        buffer = (char *)(*it);
        it_s = *(char**)  buffer;
        it_ts = *(int *) (buffer + sizeof(char*));
        if (i>=SIZE) {
            ok=false;
        } else {
            bool found = false;
            for (int j = 0; j < SIZE && !found; j++) {
                if (strcmp(s[j],it_s)) {
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

void test_string(HecubaSession& mys, const char *name) {
    IStorage* mydict = mys.createObject("dictStringKey", name);

    //Create key

    const char *s[SIZE]={"how are you",
                  "I am fine",
                  "hope you are well" };

    float lats[SIZE]={0.666, 0.777, 0.888};

    for (int i=0; i<SIZE; i++) {
        char * key = (char*)malloc(strlen(s[i])+1);
        memcpy(key, s[i], strlen(s[i])+1);
        mydict->setItem(&key,&lats[i]);
    }

    mydict->sync();

    //Iterate
    int i = 0;
    bool ok=true;
    char *ts;
    for(auto it = mydict->begin(); it != mydict->end(); it++) {
        ts = (char*)(*it);
        if (i>=SIZE) {
            ok=false;
        } else {
            bool found = false;
            for (int j = 0; j < SIZE && !found; j++) {
                if (strcmp(s[j],ts)) {
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
    HecubaSession s;
    std::cout<< "+ Session started"<<std::endl;

    s.loadDataModel("model_simple.yaml","model_simple.py");
    std::cout<< "+ Data Model loaded"<<std::endl;

    test_really_simple(s,"mydict");
    test_multiplekey(s, "mydictmultiplekey");
    test_string(s, "mydictString");

}
