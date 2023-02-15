#include <iostream>
#include <hecuba/HecubaSession.h>
#include <hecuba/StorageNumpy.h>

char * generateNumpyContent(const std::vector<uint32_t> &metas) {

    double *numpy=(double*)malloc( sizeof(double) * metas[0] * metas[1]);
    double *tmp = numpy;
            *tmp =1.0;
    double num = 1;
    for (int i=0; i<metas[0]; i++) {
        for (int j=0; j<metas[1]; j++) {
            *tmp = num++;
            std::cout<< "++ "<<i<<","<<j<< "=" << *tmp << std::endl;
            tmp+=1;
        }
    }
    return (char*) numpy;
}

void fnoshared(HecubaSession &s, std::string name) {
    std::cout<< " ---> Enter f " << std::endl;
    // if you want to synchronize at the end of scope. When we reach the end of
    // the scope of the object, the destructor is called automatically, which
    // causes the synchronzation with Cassanddra and the deallocation of the memory.
    std::vector<uint32_t> metas = {3,4};
    char * data= generateNumpyContent(metas);
    StorageNumpy sn(data,metas);
    s.registerObject (&sn);
    sn.make_persistent(name); 
    std::cout<< " <--- Exit f " << std::endl;
}
    
void fshared(HecubaSession &s, std::string name) {
    std::cout<< " ---> Enter f " << std::endl;
    // if you want to avoid the synchronization with Cassandra at the end of the scope
    // (this delays the destruction of the object until the 'deallocateObjects' 
    // is called or the end of the Session
    std::vector<uint32_t> metas = {3,4};
    char * data= generateNumpyContent(metas);
    StorageNumpy sn (data,metas);
    std::shared_ptr<StorageNumpy> snp = std::make_shared<StorageNumpy>(sn);
    // alternative declaration std::shared_ptr<StorageNumpy> snp = std::make_shared<StorageNumpy>(StorageNumpy(data,metas));

    s.registerObject (snp); 
    snp->make_persistent(name);
    //s.deallocateObjects(); --> number of reference of the object is 3, do not delete it
    std::cout<< " <--- Exit f " << std::endl;
}

void check_and_deallocate(HecubaSession& s, std::string numpyname) {
StorageNumpy snread;
bool failed = true;
    s.registerObject(&snread);
    try{
        snread.getByAlias(numpyname); // try to load the numpy, generates an exception if no data is found in the DB
        failed=false;
    } catch (std::exception &e) {
        std::cout<< "StorageNumpy is not stored: " << std::string (e.what()) << std::endl;
    }
    //in the case of objects that have not being destroyed, this function checks if they 
    //have end the insertion (non blocking function) and delete those objects with no pending insertions
    while (failed) {
        s.deallocateObjects(); // try to deallocate until the writes are completed and thus the getByAlias works
        try{
            snread.getByAlias(numpyname);
            std::cout<< "getByAlias works: StorageNumpy is stored " << std::endl;
            failed=false;
        } catch (std::exception &e) {
            std::cout<< "StorageNumpy is not stored: " << std::string (e.what()) << std::endl;
        }
    }
    std::cout<< " <-- Exit main " << std::endl;
}

int main() {
    std::cout<< " --> Enter main " << std::endl;
    HecubaSession s;
    std::string numpy_name_noshared("mynpnoshared");
    std::string numpy_name_shared("mynpshared");

    fnoshared(s, numpy_name_noshared);
    check_and_deallocate(s,numpy_name_noshared);
    fshared(s, numpy_name_shared);
    check_and_deallocate(s,numpy_name_shared);
}
