#include <hecuba/HecubaSession.h>
#include <hecuba/IStorage.h>
#include <iostream>

#define ROWS 4
#define COLS 3
char * generateKey(char* lat) {

    char * key = (char*) malloc (sizeof(char*));
    char * keycontent = (char*) malloc (strlen(lat)+1);
    memcpy(keycontent, lat, strlen(lat)+1);
    memcpy(key, &keycontent,sizeof(char*));
    return key;
}




int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;
    HecubaSession s;
    std::cout<< "+ Session started"<<std::endl;

    s.loadDataModel("string_model.yaml","string_model.py");
    std::cout<< "+ Data Model loaded"<<std::endl;

    // create a new StorageObject: "myclass" is the name of the class defined in model_complex.yaml and "mysim" is the name of the persistent object 
    IStorage* mydict = s.createObject("mydict", "outputDict");

	char * lat = "hey";
    char *key;
    key=generateKey(lat);
    int value=642;
    int *res;


    mydict->setItem((void *)key, &value);


    std::cout<< "+ completed: syncing"<<std::endl;

    res=(int *) mydict->getItem((void *)key);
    std::cout<< "Value got " << *res << std::endl;

    // we sync every thing before ending the process
#if 0
    mi_sn->sync();
    mydict->sync();
    myobj->sync();
#endif
}
