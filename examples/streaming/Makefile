CPPFLAGS=-g -std=c++11   -I ${HECUBA_ROOT}/include  -I ${HECUBA_ROOT}/include/hecuba  -L${HECUBA_ROOT}/lib   -lhfetch   -Wl,-rpath,${HECUBA_ROOT}/lib
OBJS= \
    producer \
	consumer_singlenumpy \
	producer_singlenumpy \
	apitest_iterator_streaming_producer\
	apitest_iterator_streaming_consumer\

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

.PHONY:apitest_iterator_streaming
apitest_iterator_streaming_producer: apitest_iterator_streaming
apitest_iterator_streaming_consumer: apitest_iterator_streaming
apitest_iterator_streaming: apitest_iterator_streaming.cpp
	g++ -o apitest_iterator_streaming  apitest_iterator_streaming.cpp    ${CPPFLAGS}
	cp apitest_iterator_streaming apitest_iterator_streaming_producer
	cp apitest_iterator_streaming apitest_iterator_streaming_consumer
	rm apitest_iterator_streaming

clean:
	rm ${OBJS}
