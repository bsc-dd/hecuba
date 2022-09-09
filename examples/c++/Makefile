CPPFLAGS=-g -std=c++11   -I ${HECUBA_ROOT}/include  -L${HECUBA_ROOT}/lib   -lhfetch   -Wl,-rpath,${HECUBA_ROOT}/lib
OBJS= \
    apitest_complex \
    apitest_complex_rw \
    apitest_iterator \
# DO NOT REMOVE THIS LINE (MARKS THE END OF THE 'OBJS' VARIABLE)

all: ${HECUBA_ROOT}/lib/libhfetch.so ${OBJS}

${HECUBA_ROOT}/lib/libhfetch.so:
	@echo "HECUBA_ROOT must be set"
	@false

apitest_complex: apitest_complex.cpp
	g++ -o apitest_complex   apitest_complex.cpp    ${CPPFLAGS}

apitest_complex_rw: apitest_complex_rw.cpp
	g++ -o apitest_complex_rw   apitest_complex_rw.cpp    ${CPPFLAGS}

apitest_iterator: apitest_iterator.cpp
	g++ -o apitest_iterator   apitest_iterator.cpp    ${CPPFLAGS}

clean:
	rm ${OBJS}
