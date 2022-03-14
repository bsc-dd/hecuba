CFLAGS=-g -std=c++11   -I ${HECUBA_ROOT}/include   -L${HECUBA_ROOT}/lib   -lhfetch   -Wl,-rpath,${HECUBA_ROOT}/lib
OBJS= apitest_complex

all: ${HECUBA_ROOT}/lib/libhfetch.so ${OBJS}

${HECUBA_ROOT}/lib/libhfetch.so:
	@echo "HECUBA_ROOT must be set"
	@false

apitest_complex: apitest_complex.cpp
	g++ -o apitest_complex   apitest_complex.cpp    ${CFLAGS} 

clean:
	rm ${OBJS}
