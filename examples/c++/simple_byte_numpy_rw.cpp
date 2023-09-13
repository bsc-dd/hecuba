#include <iostream>

#include <StorageNumpy.h>

#define COLS 3
#define ROWS 4


char * generateNumpyContent(const std::vector<uint32_t> &metas) {

    char *numpy=(char*)malloc(sizeof(char)*metas[0]*metas[1]);
    char *tmp = numpy;
    char num = 'a';
    for (int i=0; i<metas[0]; i++) {
        *tmp = (num++ % ('z'+1));
        std::cout<< "++ "<<i<< " ==> " << (*tmp) << std::endl;
        tmp+=1;
    }
    return (char*) numpy;
}

bool equalsNumpy(const StorageNumpy& s, char *data, std::vector<uint32_t> metas) {
    // Check metas
    if (s.metas.size() != metas.size()) {
        std::cout<< "++ dimensions differs! "<< s.metas.size()<< " != " << metas.size() << std::endl;
        return false;
    }
    for (int i=0; i<metas.size(); i++) {
        if (s.metas[i] != metas[i]) {
            std::cout<< "++ dimension "<<i<<" differs! "<< s.metas[i]<< " != " << metas[i] << std::endl;
            return false;
        }
    }
    // Check Content
    char* p = (char*)s.data;
    char* q = data;
    for (int i=0; i<metas[0]; i++) {
            if (*p != *q) {
                return false;
            }
            p++;
            q++;
    }
    return true;
}


void test_retrieve_simple(const char *name) {
    std::vector<uint32_t> metadata = {12};
    char *data = generateNumpyContent(metadata);
    StorageNumpy sn(data,metadata,'i'); // 'b' is for BOOLEAN not bytes

    sn.make_persistent(name);

    sn.sync();

    StorageNumpy sn2;
    sn2.getByAlias(name);

    if (!equalsNumpy(sn2, (char*)data, metadata)) {
       std::cout << "FAILED Retrieved Numpy ["<< name<< "] contains unexpected content (differnt from stored). " <<std::endl;
    } else {
       std::cout << "SUCCESS Retrieved Numpy ["<< name<< "] " <<std::endl;
    }
}


void test_retrieve_from_python(const char *name) {
    StorageNumpy sn;
    sn.getByAlias(name);
    char* p = (char*)sn.data;


    for (int i=0; i<sn.metas[0]; i++) {
            std::cout << "Retrieved " << i << " " <<*p<<std::endl;
            p++;
    }
}

int main() {
    std::cout<< "+ STARTING C++ APP"<<std::endl;

    std::cout << "Starting test 1 " <<std::endl;
    test_retrieve_simple("mynumpybytetoread");
    //test_retrieve_from_python("python"); // asumes there is a previously stored sn named python. To test that we can read somethin stored from python

    std::cout << "End tests " <<std::endl;
}
