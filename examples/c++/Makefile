CPPFLAGS=-g -std=c++11   -I ${HECUBA_ROOT}/include  -I ${HECUBA_ROOT}/include/hecuba  -L${HECUBA_ROOT}/lib   -lhfetch   -Wl,-rpath,${HECUBA_ROOT}/lib

OBJS= \
    apitest_iterator \
    simple_numpy \
    simple_numpy_rw \
    simple_dict_rw \
	nested_rw \
	test_shared_ptr \
	parent_code \
	simple_so \
	very_simple_so \
	nested_so \

# DO NOT REMOVE THIS LINE (MARKS THE END OF THE 'OBJS' VARIABLE)

all: ${HECUBA_ROOT}/lib/libhfetch.so ${OBJS}

${HECUBA_ROOT}/lib/libhfetch.so:
	@echo "HECUBA_ROOT must be set"
	@false

apitest_iterator: apitest_iterator.cpp
	g++ -o $@   $^    ${CPPFLAGS}

simple_numpy: simple_numpy.cpp
	g++ -o $@ $^   ${CPPFLAGS}

simple_numpy_rw: simple_numpy_rw.cpp
	g++ -o $@ $^   ${CPPFLAGS}

simple_dict_rw: simple_dict_rw.cpp
	g++ -o $@ $^   ${CPPFLAGS}

nested_rw: nested_rw.cpp
	g++ -o $@ $^   ${CPPFLAGS}

test_shared_ptr: test_shared_ptr.cpp
	g++ -o $@ $^   ${CPPFLAGS}
parent_code: parent_code.cpp
	g++ -o $@ $^   ${CPPFLAGS}

simple_so: simple_so.cpp
	g++ -o $@ $^   ${CPPFLAGS}

very_simple_so: very_simple_so.cpp
	g++ -o $@ $^   ${CPPFLAGS}

nested_so: nested_so.cpp
	g++ -o $@ $^   ${CPPFLAGS}
clean:
	rm ${OBJS}
