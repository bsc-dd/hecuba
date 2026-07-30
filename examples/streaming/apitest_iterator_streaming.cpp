#include <iostream>

#include <StorageDict.h>
#include <StorageStream.h>
#include <KeyClass.h>
#include <ValueClass.h>
#define SIZE 3
#define ROWS 3
#define COLS 4

using namespace Hecuba;

std::string id;

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

using IntKeyClass = KeyClass<int32_t>;
using NumpyValueClass = ValueClass<StorageNumpy>;

class DictWithNumpy: public StorageDict <IntKeyClass, NumpyValueClass, DictWithNumpy>, public StorageStream {

};

char * generateNumpyContent(const std::vector<uint32_t> &metas, int from = 1) {

    double *numpy=(double*)malloc(sizeof(double)*metas[0]*metas[1]);
    double *tmp = numpy;
    double num = from;
    for (int i=0; i<metas[0]; i++) {
        for (int j=0; j<metas[1]; j++) {
            std::cout<< "++ "<<i<<","<<j<<std::endl;
            *tmp = num++;
            tmp+=1;
        }
    }
    std::cout<< "+ Generated NUMPY ["<<metas[0]<<", "<<metas[1]<<"] using "<<sizeof(double)*metas[0]*metas[1]<<"bytes at "<<std::hex<<(void*)numpy<<std::endl;
    return (char*) numpy;
}

void test_dict_with_numpy (const char *dictName, int producer) {

	DictWithNumpy mydict;
    std::string numpy_name;
    IntKeyClass key;

    std::vector<uint32_t> metadata = {ROWS, COLS};

    if (producer) {
	mydict.make_persistent(dictName );
    std::cout<< "+ Dictionary "<<dictName<< " object created"<<std::endl;

    // create a StorageNumpy and then add it to the StorageDict

    for (int i = 0; i< 3 ; i++) {
        char* data = generateNumpyContent(metadata,(i*metadata[0]*metadata[1])+1); // dummy calculation for the numbers of the numpys
        std::cout<< "+ value created at "<<std::hex<<(void*)data<<std::endl;
        // createObject executes a 'new', therefore reference MUST be deleted by the user
        StorageNumpy my_sn(data, metadata);
        numpy_name=std::string(dictName)+std::string("sn");
        numpy_name+=i;
	    my_sn.make_persistent(numpy_name.c_str());
        std::cout<< "+  numpy persisted"<< numpy_name << std::endl;

	    key = IntKeyClass(42+i);
        std::cout<< "+  key created"<< std::endl;

	    //value = NumpyValueClass(my_sn);
        NumpyValueClass value(my_sn);
        std::cout<< "+ value created at "<<std::hex<<(void*)my_sn.getStorageID()<<std::endl;
        mydict[key] = value;

        std::cout<< "+ mydict setitem completed "<<std::endl;
    }
    return;
    }
    // CONSUMER
	mydict.getByAlias(dictName );
    std::cout<< "+ mydict getByAlias completed "<<std::endl;


    int nit = 0;
    bool ok = true;
    for(auto it = mydict.begin(); it != mydict.end(); it++, nit++) {

        std::cout<< "+ Starting iteration "<< nit << std::endl;
        key = it->first;
        NumpyValueClass value = it->second;
        StorageNumpy &sn_rcv=NumpyValueClass::get<0>(value);
        int32_t key_rcv=IntKeyClass::get<0>(key);

        //check received values are the expected
        if (key_rcv != 42+nit) {
            std::cout << "key "<< nit << "received is wrong: got " << key_rcv<< " expected " << 42+nit << std::endl;
            ok=false;
        }
        double * data = (double *) generateNumpyContent(metadata,(nit*metadata[0]*metadata[1])+1); // dummy calculation for the numbers of the numpys
        double *psn = (double *)sn_rcv.data;
        for (int i = 0; i< metadata[0]; i++) {
            for (int j = 0; j < metadata[1]; j++) {
                    if (*psn != *data) {
                        std::cout << "value "<< nit << "received is wrong: got " << *psn<< " expected " << *data << std::endl;
                        ok = false;
                    }
                    psn++;
                    data++;
            }
        }
    }
    if (ok) {
        std::cout<< "+ End test without errors" << std::endl;
    } else {
        std::cout<< "+ Test FAILED" << std::endl;

    }

}
void test_dict_with_numpy_poll (const char *dictName, int producer) {

	DictWithNumpy mydict;
    std::string numpy_name;
    IntKeyClass key;

    std::vector<uint32_t> metadata = {ROWS, COLS};

    if (producer) {
	mydict.make_persistent(dictName );
    std::cout<< "+ Dictionary "<<dictName<< " object created"<<std::endl;

    // create a StorageNumpy and then add it to the StorageDict

    for (int i = 0; i< 3 ; i++) {
        char* data = generateNumpyContent(metadata,(i*metadata[0]*metadata[1])+1); // dummy calculation for the numbers of the numpys
        std::cout<< "+ value created at "<<std::hex<<(void*)data<<std::endl;
        // createObject executes a 'new', therefore reference MUST be deleted by the user
        StorageNumpy my_sn(data, metadata);
        numpy_name=std::string(dictName)+std::string("sn");
        numpy_name+=i;
	    my_sn.make_persistent(numpy_name.c_str());
        std::cout<< "+  numpy persisted"<< numpy_name << std::endl;

	    key = IntKeyClass(42+i);
        std::cout<< "+  key created"<< std::endl;

	    //value = NumpyValueClass(my_sn);
        NumpyValueClass value(my_sn);
        std::cout<< "+ value created at "<<std::hex<<(void*)my_sn.getStorageID()<<std::endl;
        mydict[key] = value;

        std::cout<< "+ mydict setitem completed "<<std::endl;
    }
    return;
    }
    // CONSUMER
	mydict.getByAlias(dictName );
    std::cout<< "+ mydict getByAlias completed "<<std::endl;


    int nit = 0;
    bool ok = true;
    for(int nit = 0 ; nit <3;  nit++) {

        std::cout<< "+ Starting iteration "<< nit << std::endl;
        auto it = mydict.poll();
        key = it->first;
        NumpyValueClass value = it->second;
        StorageNumpy &sn_rcv=NumpyValueClass::get<0>(value);
        int32_t key_rcv=IntKeyClass::get<0>(key);

        //check received values are the expected
        if (key_rcv != 42+nit) {
            std::cout << "key "<< nit << "received is wrong: got " << key_rcv<< " expected " << 42+nit << std::endl;
            ok=false;
        }
        double * data = (double *) generateNumpyContent(metadata,(nit*metadata[0]*metadata[1])+1); // dummy calculation for the numbers of the numpys
        double *psn = (double *)sn_rcv.data;
        for (int i = 0; i< metadata[0]; i++) {
            for (int j = 0; j < metadata[1]; j++) {
                    if (*psn != *data) {
                        std::cout << "value "<< nit << "received is wrong: got " << *psn<< " expected " << *data << std::endl;
                        ok = false;
                    }
                    psn++;
                    data++;
            }
        }
    }
    if (ok) {
        std::cout<< "+ End test without errors" << std::endl;
    } else {
        std::cout<< "+ Test FAILED" << std::endl;

    }

}

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
                //std::cout << " test_string s["<<j<<"] = "<<s[j]<<" == "<<ts<<" = ts"<<std::endl;
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
    if (argc>1) {

        // There is a problem when consecutive instances of the same program
        // are launched one after the other, as the name of the persistent
        // objects (and therefore their UUIDs) are also the same. This is
        // problematic in multiple ways. One of the problems is that KAFKA
        // creates and destroys the same topics (UUIDs)... but as the group.id
        // is the same then it does not behave correctly :( (mainly the first
        // instance works, but the second receives a dummy value at the
        // `poll`... which tries to instantiate and fails).
        // Add this `id` argument to diferentiate different program
        // invocations.
        //
        id = std::string(argv[1]);
    }

    std::cout<< "+ STARTING C++ APP"<<std::endl;
    std::cout<< "+ Session started"<<std::endl;

    std::cout << "Starting test 1 " <<std::endl;
    test_really_simple("mydict", producer);

    std::cout << "Starting test 2 " <<std::endl;
    test_multiplekey("mydictmultiplekey",producer);

    std::cout << "Starting test 3 " <<std::endl;
    test_string("mydictString", producer);

    std::cout << "Starting test 4 " <<std::endl;
    test_dict_with_numpy((id + "dictWithNumpy_iterator").c_str(), producer);

    std::cout << "Starting test 5 " <<std::endl;
    test_dict_with_numpy_poll((id + "dictWithNumpy_poll").c_str(), producer);

    std::cout << "End tests " <<std::endl;
}
