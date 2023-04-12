#include <StorageObject.h>

class innerSO: public StorageObject {
    public:
    HECUBA_ATTRS(
            int, a,
            float, b
           )
};

class externSO: public StorageObject {
    public:
    HECUBA_ATTRS(
            innerSO, inner,
            int, attr1,
            std::string, attr2
            )
};

void test_assign_nested(std::string extern_name, std::string inner_name) {
    innerSO o;
    o.make_persistent(inner_name);
    o.a = 42;
    o.b = (float)3.14;

    externSO p;
    p.make_persistent(extern_name);

    p.attr1=666;
    p.attr2="hi";
    p.inner = o;
    p.sync();
}

void test_retrieve_constructor_operator (std::string extern_name) {
    externSO retrieve;
    retrieve.getByAlias(extern_name);
    if (retrieve.attr1 != 666) {
        std::cout << "Retrieved attr1 is "<< retrieve.attr1<< " and should be 666: FAILED" << std::endl;
    } else {
        std::cout << "Assigned attr1 is "<< retrieve.attr1 << ": PASSED" << std::endl;

    }
    std::string mystring(retrieve.attr2);

    if (mystring != "hi") {
        std::cout << "Retrieved attr2 is "<< mystring << " and should be 'hi': FAILED" << std::endl;
    } else {
        std::cout << "Assigned attr2 is "<< mystring << ": PASSED" << std::endl;
    }
    
    // invokes copy constructor of innerSO
    innerSO retrieve_inner = retrieve.inner;
    if ((int)retrieve_inner.a != 42) { 
        std::cout << "Retrieved inner.a is "<< (int)retrieve_inner.a << " and should be 42: FAILED" << std::endl;

    } else {
        std::cout << "Assigned inner.a is "<< (int)retrieve_inner.a << ": PASSED" << std::endl;

    }
    if (retrieve_inner.b != (float)3.14) {
        std::cout << "Retrieved inner.b is "<< (float) retrieve_inner.b << " and should be 3.14: FAILED" << std::endl;

    } else {
        std::cout << "Assigned inner.b is "<< (float)retrieve_inner.b << ": PASSED" << std::endl;

    }

}

void test_retrieve_assign_operator (std::string extern_name) {
    externSO retrieve;
    retrieve.getByAlias(extern_name);
    if (retrieve.attr1 != 666) {
        std::cout << "Retrieved attr1 is "<< retrieve.attr1<< " and should be 666: FAILED" << std::endl;
    } else {
        std::cout << "Assigned attr1 is "<< retrieve.attr1 << ": PASSED" << std::endl;

    }
    std::string mystring(retrieve.attr2);

    if (mystring != "hi") {
        std::cout << "Retrieved attr2 is "<< mystring << " and should be 'hi': FAILED" << std::endl;
    } else {
        std::cout << "Assigned attr2 is "<< mystring << ": PASSED" << std::endl;
    }
    
    innerSO retrieve_inner;
    
    // invokes assignment operator of innerSO
    retrieve_inner= retrieve.inner;
    if ((int)retrieve_inner.a != 42) { 
        std::cout << "Retrieved inner.a is "<< (int)retrieve_inner.a << " and should be 42: FAILED" << std::endl;

    } else {
        std::cout << "Assigned inner.a is "<< (int)retrieve_inner.a << ": PASSED" << std::endl;

    }
    if (retrieve_inner.b != (float)3.14) {
        std::cout << "Retrieved inner.b is "<< (float) retrieve_inner.b << " and should be 3.14: FAILED" << std::endl;

    } else {
        std::cout << "Assigned inner.b is "<< (float)retrieve_inner.b << ": PASSED" << std::endl;

    }

}

void test_retrieve_without_temporal_variable (std::string extern_name) {
    externSO retrieve;
    retrieve.getByAlias(extern_name);

    if (retrieve.attr1 != 666) {
        std::cout << "Retrieved attr1 is "<< retrieve.attr1<< " and should be 666: FAILED" << std::endl;
    } else {
        std::cout << "Assigned attr1 is "<< retrieve.attr1 << ": PASSED" << std::endl;

    }
    std::string mystring(retrieve.attr2);

    if (mystring != "hi") {
        std::cout << "Retrieved attr2 is "<< mystring << " and should be 'hi': FAILED" << std::endl;
    } else {
        std::cout << "Assigned attr2 is "<< mystring << ": PASSED" << std::endl;
    }

    if (((innerSO)(retrieve.inner)).a != 42) { 
        std::cout << "Retrieved retrieve.inner.a is "<< ((innerSO)(retrieve.inner)).a << " and should be 42: FAILED" << std::endl;

    } else {
        std::cout << "Assigned retrieve.inner.a is "<< ((innerSO)(retrieve.inner)).a << ": PASSED" << std::endl;

    }
    if (((innerSO)(retrieve.inner)).b != (float)3.14) {
        std::cout << "Retrieved retrieve.inner.b is "<<  ((innerSO)(retrieve.inner)).b << " and should be 3.14: FAILED" << std::endl;

    } else {
        std::cout << "Assigned retrieve.inner.b is "<< ((innerSO)(retrieve.inner)).b << ": PASSED" << std::endl;

    }

}

main() {
    test_assign_nested("externSO", "innerSO");
    test_retrieve_constructor_operator("externSO");
    test_retrieve_assign_operator("externSO");
    test_retrieve_without_temporal_variable("externSO");

}




