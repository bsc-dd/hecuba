#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>


std::string UUID2str(uint64_t* c_uuid) {
    /* This MUST match with the 'cass_statement_bind_uuid' result */
    char str[37] = {};
    unsigned char* uuid = reinterpret_cast<unsigned char*>(c_uuid);
    //std::cout<< "HecubaSession: uuid2str: BEGIN "<<std::hex<<c_uuid[0]<<c_uuid[1]<<std::endl;
    sprintf(str,
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        uuid[0], uuid[1], uuid[2], uuid[3],
        uuid[4], uuid[5],
        uuid[6], uuid[7],
        uuid[8], uuid[9],
        uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]
        );
    //std::cout<< "HecubaSession: uuid2str: "<<str<<std::endl;
    return std::string(str);
}

#define ROWS 4
#define COLS 3
char * generateKey(float lat, int ts) {

    char * key = (char*) malloc (sizeof(float) + sizeof(int));
    float *lat_key = (float*) key;
    *lat_key = lat;
    int *ts_key = (int*) (key + sizeof(float));
    *ts_key = ts;
    return key;
}

char * generateMetas() {
    unsigned int * metas = (unsigned int *) malloc(sizeof(unsigned int) * 3);

    metas[0]=2; // number of dimmensions
    metas[1]=ROWS; // number of elements in the first dimmension
    metas[2]=COLS; // number of elements in the second dimmension
    //metas[3]=sizeof(double); //'f' ONLY FLOATS SUPPORTED

    return (char *) metas;
}

char * generateNumpyContent() {

    double *numpy=(double*)malloc(sizeof(double)*COLS*ROWS);
    double *tmp = numpy;
    double num = 1;
    for (int i=0; i<ROWS; i++) {
        for (int j=0; j<COLS; j++) {
            std::cout<< "++ "<<i<<","<<j<<std::endl;
            *tmp = num++;
            tmp+=1;
        }
    }
    return (char*) numpy;
}

void test_simpleDict(HecubaSession& s, const char *name) {
    IStorage* simpledict = s.createObject("simpledict", name);
    int keyInt = 666;
    int valueInt = 777;
    simpledict->setItem(&keyInt, &valueInt);

    bool passed = true;
    int value_retrieved;
    simpledict->getItem(&keyInt, (void*) &value_retrieved);
    if ( value_retrieved != valueInt ){
        std::cout<< "ERROR getting item "<< keyInt <<" from object " << name << ": expected " << std::to_string(valueInt) << ", received " << std::to_string(value_retrieved) << std::endl;
        passed = false;
    }

    if (passed) {
        std::cout << " TEST: test_simpleDict PASSED"<< std::endl;
    } else {
        std::cout << " TEST: test_simpleDict FAILED"<< std::endl;
    }
}

void test_storageobj2attr(HecubaSession& s, const char *name) {
    IStorage* myobj2 = s.createObject("info", name);
    int total_ts = 10000;
    myobj2->setAttr("total_ts", &total_ts);
    int output_freq=10;
    myobj2->setAttr("output_freq", &output_freq);

    int value_retrieved;
    bool passed = true;
    myobj2->getAttr("total_ts", (void*) &value_retrieved);
    if ( value_retrieved != total_ts ){
        std::cout<< "ERROR getting integer attribute total_ts from object " << name << ": expected " << std::to_string(total_ts) << ", received " << std::to_string(value_retrieved) << std::endl;
        passed = false;
    }

    myobj2->getAttr("output_freq", (void*) &value_retrieved);
    if ( value_retrieved != output_freq ){
        std::cout<< "ERROR getting integer attribute output_freq from object " << name <<": expected " << std::to_string(output_freq) << ", received " << std::to_string(value_retrieved) << std::endl;
        passed = false;
    }

    if (passed) {
        std::cout << " TEST: test_storageobj2attr PASSED"<< std::endl;
    } else {
        std::cout << " TEST: test_storageobj2attr FAILED"<< std::endl;
    }

}

IStorage * createNumpy(HecubaSession& s, const char *name) {
    char * numpymeta;
    numpymeta = generateMetas();
    char *valueNP = generateNumpyContent();
    IStorage *my_sn = s.createObject("hecuba.hnumpy.StorageNumpy", name, numpymeta, valueNP);
    return my_sn;
}

void test_storageobjNumpy(HecubaSession& s, const char *name) {
    IStorage* sonumpy = s.createObject("sonumpy", name);

    IStorage* my_sn = createNumpy(s, (std::string(name) + "minp").c_str());
    char *valueNP = (char*) my_sn->getNumpyData();

    sonumpy->setAttr("num", my_sn);
    my_sn->sync(); // WARNING: We are about to read the data and we need to ensure the data is in Cassandra

    bool passed = true;
    IStorage* value_retrieved;
    sonumpy->getAttr("num", &value_retrieved);
    void* data = value_retrieved->getNumpyData();
    for (int i=0;i<ROWS*COLS;i++) {
        if (((double*)data)[i] != ((double*)valueNP)[i]) {
            std::cout<< "ERROR getting numpy from SO " << name <<" at attribute 'num' at position "<< std::to_string(i) << " expected " << std::to_string(((double*)valueNP)[i]) << ", received " << std::to_string(((double*)data)[i]) << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << " TEST: test_storageobjNumpy PASSED"<< std::endl;
    } else {
        std::cout << " TEST: test_storageobjNumpy FAILED"<< std::endl;
    }

}

void test_storageobjComplex(HecubaSession& s, const char *name) {
    IStorage* myclass = s.createObject("myclass", name);

    const char *sim_id = "id";
    myclass->setAttr("sim_id", &sim_id);

    IStorage* info = s.createObject("info", (std::string(name)+"sim_info").c_str());
    int total_ts = 10000;
    info->setAttr("total_ts", &total_ts);
    int output_freq=10;
    info->setAttr("output_freq", &output_freq);

    myclass->setAttr("sim_info", info);

    IStorage *my_sn = createNumpy(s, (std::string(name) + "mynp").c_str());
    char* valueNP = (char*) my_sn->getNumpyData();
	float lat = 0.666;
    int ts = 42;
    char* key = generateKey(lat, ts);

    IStorage *metrics = s.createObject("metrics", (std::string(name) + "mymetrics").c_str());
    metrics->setItem(key, my_sn);

    myclass->setAttr("submetrics", metrics);

    IStorage *simpledict = s.createObject("simpledict", (std::string(name) + "mysimple").c_str());
    int keyInt = 666;
    int valueInt = 42;
    simpledict->setItem(&keyInt, &valueInt);

    myclass->setAttr("simple", simpledict);

    bool passed = true;
    char* str_retrieved;
    myclass->getAttr("sim_id", &str_retrieved);

    if (std::string(str_retrieved).compare(sim_id)!=0) {
        passed = false;
        std::cout<< "ERROR getting attribute 'sim_id' from SO " << name <<" expected " << sim_id << ", received " << str_retrieved << std::endl;
    }


    IStorage* value_retrieved;
    myclass->getAttr("simple", &value_retrieved);

    if (UUID2str(value_retrieved->getStorageID()).compare(UUID2str(simpledict->getStorageID())) != 0) {
        passed = false;
        std::cout<< "ERROR getting attribute 'simple' from SO " << name <<" expected " << UUID2str(simpledict->getStorageID()) << ", received " << UUID2str(value_retrieved->getStorageID())  << std::endl;
    }

    if (passed) {
        std::cout << " TEST: test_storageobjComplex PASSED"<< std::endl;
    } else {
        std::cout << " TEST: test_storageobjComplex FAILED"<< std::endl;
    }
}

void test_dictmultikey_and_numpy(HecubaSession& s, const char *name) {
    IStorage* metrics = s.createObject("metrics", name);

	float lat = 0.666;
    int ts = 42;
    char *key;
    key=generateKey(lat,ts);

    IStorage *my_sn = createNumpy(s, (std::string(name) + "minp").c_str());
    char *valueNP = (char*) my_sn->getNumpyData();

    metrics->setItem(key, my_sn);
    my_sn->sync(); // WARNING: We are about to read the data and we need to ensure the data is in Cassandra

    bool passed = true;
    IStorage* mvalues;
    metrics->getItem(key,&mvalues);
    std::cout<< "----- Retrieving NUMPY DATA from object"<<std::endl;
    void* data = mvalues->getNumpyData();
    std::cout<< "----- Printing NUMPY DATA "<<std::endl;
    for (int i=0;i<ROWS*COLS;i++) {
        if (((double*)data)[i] != ((double*)valueNP)[i]) {
            std::cout<< "ERROR getting numpy from dictionary " << name <<", key (0.666,42) at position "<< std::to_string(i) << " expected " << std::to_string(((double*)valueNP)[i]) << ", received " << std::to_string(((double*)data)[i]) << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << " TEST: test_dictmultikey_and_numpy PASSED"<< std::endl;
    } else {
        std::cout << " TEST: test_dictmultikey_and_numpy FAILED"<< std::endl;
    }

}

void test_dictMultiKeyMultiValue(HecubaSession&s, const char *name) {
    // CREATING A STORAGEDICT WITH MULTIVALUE CONTAINING ALL THE OTHER SUBOBJECTS
    IStorage* mydictmulti = s.createObject("dictMultiValue", name);

    IStorage* info = s.createObject("info", (std::string(name)+"sim_info").c_str());
    int total_ts = 10000;
    info->setAttr("total_ts", &total_ts);
    int output_freq=10;
    info->setAttr("output_freq", &output_freq);

    IStorage *my_sn = createNumpy(s, (std::string(name) + "mynp").c_str());

	float lat = 0.666;
    int ts = 42;
    char* key = generateKey(lat, ts);

    IStorage *metrics = s.createObject("metrics", (std::string(name) + "mymetrics").c_str());
    metrics->setItem(key, my_sn);
    my_sn->sync(); // WARNING: We are about to read the data and we need to ensure the data is in Cassandra

    /// Prepare buffer for values
    void *buffer = malloc( sizeof(char*)   //mystr
            +sizeof(int)  //num
            +sizeof(char*)    //obj
            +sizeof(char*)  //numpy
            +sizeof(char*)  //dict
    );
    char *p =(char*)buffer;
    //std::cout<< "++ CREATING OBJECT DICT WITH NUMPY: setting the attribute mystr"<<std::endl;
    const char *mystr="hello holidays";
    memcpy(p, &mystr,sizeof(char*));
    p+=sizeof(char*);
    //std::cout<< "++ CREATING OBJECT DICT WITH NUMPY: setting the attribute num"<<std::endl;
    uint32_t num = 43;
    memcpy(p, &num,sizeof(int));
    p+=sizeof(int);
    //std::cout<< "++ CREATING OBJECT DICT WITH NUMPY: setting the attribute obj"<<std::endl;
    memcpy(p, &info,sizeof(char*));
    p+=sizeof(char*);
    //std::cout<< "++ CREATING OBJECT DICT WITH NUMPY: setting the attribute numpy"<<std::endl;
    memcpy(p, &my_sn,sizeof(char*));
    p+=sizeof(char*);
    //std::cout<< "++ CREATING OBJECT DICT WITH NUMPY: setting the attribute dict"<<std::endl;
    memcpy(p, &metrics,sizeof(char*));
    p+=sizeof(char*);

    /// Prepare buffer for key
    char* mykey = (char*) malloc(
             sizeof(char*)   //key1
            +sizeof(int)    //key2
    );
    uint64_t offset = 0;
    const char * key1 = "hiworld";
    memcpy(mykey+offset, &key1,sizeof(char*));
    offset+=sizeof(char*);
    int key2 = 66642;
    memcpy(mykey+offset, &key2,sizeof(int));

    mydictmulti->setItem(mykey, buffer);

    char *value_retrieved;
    mydictmulti->getItem(mykey, (void*)&value_retrieved);
    offset = 0;

    // Decode multivalue...
    char *mystr2 = *(char**)(value_retrieved+offset);
    offset += sizeof(char*);
    if (strcmp(mystr2, "hello holidays")!=0) {
        throw ModuleException(" Attribute 'mystr' contains " + std::string(mystr2) + " that does not match "+std::string("hello holidays"));
    }
    //std::cout<< "++ INSTANTIATING OBJECT DICT WITH NUMPY: getting the attribute mystr"<<mystr2<<std::endl;

    uint32_t num2 = *(int*)(value_retrieved+offset);
    offset+=sizeof(int);
    if (num2 != num) {
        throw ModuleException(" Attribute 'num' contains " + std::to_string(num2) + " that does not match "+std::to_string(num));
    }

    std::cout<< "++ INSTANTIATING OBJECT DICT WITH NUMPY: getting the attribute num"<<num2<<std::endl;

    bool passed=true;
    IStorage *myobj = *(IStorage **)(value_retrieved+offset);
    if (UUID2str(myobj->getStorageID()).compare(UUID2str(info->getStorageID())) != 0) {
        passed = false;
        std::cout<< "ERROR getting item from Dictionary " << name <<" expected " << UUID2str(info->getStorageID()) << ", received " << UUID2str(myobj->getStorageID())  << std::endl;
    }


    offset += sizeof(char*);
    myobj = *(IStorage **)(value_retrieved+offset);
    if (UUID2str(myobj->getStorageID()).compare(UUID2str(my_sn->getStorageID())) != 0) {
        passed = false;
        std::cout<< "ERROR getting item from Dictionary " << name <<" expected " << UUID2str(my_sn->getStorageID()) << ", received " << UUID2str(myobj->getStorageID())  << std::endl;
    }

    offset += sizeof(char*);
    myobj = *(IStorage **)(value_retrieved+offset);
    if (UUID2str(myobj->getStorageID()).compare(UUID2str(metrics->getStorageID())) != 0) {
        passed = false;
        std::cout<< "ERROR getting item from Dictionary " << name <<" expected " << UUID2str(metrics->getStorageID()) << ", received " << UUID2str(myobj->getStorageID())  << std::endl;
    }

    if (passed) {
        std::cout << " TEST: test_dictMultiKeyMultiValue PASSED"<< std::endl;
    } else {
        std::cout << " TEST: test_dictMultiKeyMultiValue FAILED"<< std::endl;
    }

    delete(my_sn);
    delete(metrics);
    delete(info);
    delete(mydictmulti);// We cannot end the execution before the inserts into cassandra ends
}

int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;
    HecubaSession s;
    std::cout<< "+ Session started"<<std::endl;

    s.loadDataModel("model_complex.yaml","model_complex.py");
    std::cout<< "+ Data Model loaded"<<std::endl;

    test_dictMultiKeyMultiValue(s, "t_dictmultikeymultival");
    test_storageobj2attr(s, "t_storageobj2attr");
    test_simpleDict(s, "t_simpleDict");
    test_dictmultikey_and_numpy(s, "t_dictmultikey_and_numpy");
    test_storageobjNumpy(s, "t_sonumpy");
    test_storageobjComplex(s, "t_socomplex");



}
