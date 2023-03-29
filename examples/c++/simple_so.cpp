#include <hecuba/HecubaSession.h>
#include <hecuba/StorageObject.h>

class mySO: public StorageObject {
    public:
    HECUBA_ATTRS(
            int, a,
            float, b,
            std::string, c
           )
};

void test_simple_rw(const char* name) {
    mySO o;
    o.make_persistent(name);
    o.a = 42;
    o.b = (float)3.14;
    o.c = std::string("hi");

    o.sync();

}

void test_simple_retrieve(const char* name) {
    mySO retrieve;
    retrieve.getByAlias(name);
    if ((int)retrieve.a != 42) {
        std::cout << "Retrieved a is "<< (int)retrieve.a << " and should be 42: FAILED" << std::endl;
    } else {
        std::cout << "Assigned a is "<< (int)retrieve.a << ": PASSED" << std::endl;

    }
    if ((float)retrieve.b != (float)3.14) {
        std::cout << "Retrieved b is "<< (float)retrieve.b << " and should be 3.14" << std::endl;
    } else {
        std::cout << "Assigned b is "<< (float)retrieve.b << ": PASSED" << std::endl;
    }

    if ((std::string)retrieve.c != "hi") {
        std::cout << "Retrieved c is "<< (std::string)retrieve.c << " and should be 'hi'" << std::endl;
    } else {
        std::cout << "Assigned c is "<< (std::string)retrieve.c << ": PASSED" << std::endl;
    }
}

void test_assignment(const char * namesrc, const char* namedst) {
    mySO o;
    o.make_persistent(namesrc);
    o.a = 42;
    o.b = (float)3.14;
    o.c = "hi";
    o.sync();
    mySO q;
    q.make_persistent(namedst);
    q.a = o.a;   // SO_Attribute = SO_Attribute
    q.b = o.b;
    q.c = o.c;

    if ((int) q.a != 42) {
        std::cout << "Assigned a is "<< (int)q.a << " and should be 42: FAILED" << std::endl;
    } else {
        std::cout << "Assigned a is "<< (int)q.a << ": PASSED" << std::endl;
    }

    if ((float) q.b != (float)3.14) {
        std::cout << "Assigned b is "<< (float)q.a << " and should be 3.14: FAILED" << std::endl;
    } else {
        std::cout << "Assigned b is "<< (float)q.b << ": PASSED" << std::endl;
    }

    if ((std::string)q.c != "hi") {
        std::cout << "Assigned c is "<< (std::string)q.c << " and should be 'hi'" << std::endl;
    } else {
        std::cout << "Assigned c is "<< (std::string)q.c << ": PASSED" << std::endl;
    }
}

main() {
    test_simple_rw("testSO");
    test_assignment("srcSO", "dstSO");
    test_simple_retrieve("testSO");

}

