CPPFLAGS=-g -std=c++11   -I ${HECUBA_ROOT}/include  -I ${HECUBA_ROOT}/include/hecuba  -L${HECUBA_ROOT}/lib   -lhfetch   -Wl,-rpath,${HECUBA_ROOT}/lib
OBJS= \
    producer \
	consumer_singlenumpy \
	producer_singlenumpy \

# DO NOT REMOVE THIS LINE (MARKS THE END OF THE 'OBJS' VARIABLE)

all: ${HECUBA_ROOT}/lib/libhfetch.so ${OBJS}

${HECUBA_ROOT}/lib/libhfetch.so:
	@echo "HECUBA_ROOT must be set"
	@false

producer: producer.cpp
	@echo ==========================================================================
	@echo 'Producer' application expects 'consumer.py' to be run to receive the data
	@echo ==========================================================================
	g++ -o producer  producer.cpp    ${CPPFLAGS}

consumer_singlenumpy: consumer_singlenumpy.cpp
	g++ -o consumer_singlenumpy  consumer_singlenumpy.cpp    ${CPPFLAGS}

producer_singlenumpy: producer_singlenumpy.cpp
	g++ -o producer_singlenumpy  producer_singlenumpy.cpp    ${CPPFLAGS}

clean:
	rm ${OBJS}
