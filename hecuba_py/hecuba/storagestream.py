from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import Producer, Consumer, TopicPartition
from . import config, log
import pickle

#requirement: only support persistent StorageDicts
class StorageStream(object):

    @staticmethod
    def _propagate_stream_capability(self, topic_name=None):
        # Set the stream attributes into 'self' instance
        # WARNING: This is done also from other Class instances (StorageNumpy, StorageDict, StorageObj)
        self._kafka_names = str.join(",", config.kafka_names)
        self.stream_enabled=True
        if topic_name is not None:
            self._topic_name = str(topic_name)
            self._hcache.enable_stream(self._topic_name, {'kafka_names': str.join(",",config.kafka_names)})
            log.debug("Enabling STREAM %s", self._topic_name)
        else:
            self._topic_name = None

        # Producer and Consumer are lazily created only if 'setitem' or 'poll' are invoked.
        self._stream_producer_enabled = False
        self._stream_consumer_enabled = False

    @staticmethod
    def enable_stream(self, topic_name):
        self._topic_name = topic_name
        self._hcache.enable_stream(self._topic_name, {'kafka_names': str.join(",",config.kafka_names)})
        log.debug("Enabling STREAM %s", self._topic_name)

    def __init__(self, *args, topic_name=None,  **kwargs):
        # we support both type of class definitions: class myclass(StorageDict,StorageStream) and class (StorageStream, StorageDict)
        # In the first case the constructor of StorageDict is called first and thus here it can initialize the streaming properly (with the topic_name)
        # In the second case we delay the initialization of the storagestream until the first setitem is performed (or the make_persistent for volatil SD)
        log.debug("StorageStream: INITIALIZE ")
        super(StorageStream, self).__init__(*args, **kwargs)


# d[42]=666 --> __setitem__

#    def __send_values_kafka(self, key, val):
#        if not self._stream_producer_enabled:
#            self._hcache.enable_stream_producer()
#            self._stream_producer_enabled=True
#        tosend=[]
#        if not isinstance(val,list):
#            val = [val]
#
#        for element in val:
#            if isinstance(element, IStorage):
#                tosend.append(element.storage_id)
#                #diferenciar caso numpy,  de embedded set(TODO), de storageobj
#                element.send(element.storage_id, element.data) # FIXME
#            else:
#                tosend.append(element)
#
#        self._hcache.send_event(key, tosend) # Send list with storageids and basic_types


#    def __setitem__(self,key, value):
#        log.debug("StorageStream: SETITEM sid {} key {} value {}".format(self.storage_id, key, value))
#        if not self._stream_producer_enabled:
#            self._hcache.enable_stream_producer()
#            self._stream_producer_enabled=True
#        super().__setitem__(key,value)
#        # In a wonderful world 'setitem' would return the 'modified' values... but here it does not
#        modifiedvalues = super().__getitem__(key)
#        self.__send_values_kafka(modifiedvalues);
#        para cada elt de value
#            si elt es istorage
#                elt->send(valor de elt sea lo que sea)
#        self.send(lista de storage_ids y valores que no son istorage)<-------
#
#    def send(self, key, value):
#        log.debug("StorageStream: SEND sid {} key {} value {}".format(self.storage_id, key, value))
#        self.__setitem__(key,value)
#
#
#    def poll(self):
#        log.debug("StorageStream: POLL sid {} ".format(self.storage_id))
#        import time
#        if not self._stream_consumer_enabled:
#            self._hcache.enable_stream_consumer()
#            self._stream_consumer_enabled=True
#        start = time.time()
#        row = self._hcache.poll()
#        stop = time.time()
#
#        print("poll row type {} got row: {}".format(type(row),row),flush=True)
#        v=row[-(len(row)-self._k_size)]
#        k=row[0:self._k_size]
#        print("poll row type {} got row: {}".format(type(row),row),flush=True)
#        print("poll k {} v {}".format(k,v),flush=True)
#        #v = super().__getitem__(key)
#        self._hcache.add_to_cache(self._make_key(k),self._make_value(v)) #this line only works for StorageDicts
#        stop2 = time.time()
#        print("TIME POLL {} TIME GETITEM {}".format(stop-start, stop2-stop), flush=True)
#        para cada elt de value
#            si elt es istorage
#                elt->send(valor de elt sea lo que sea)
#        self.send(lista de storage_ids y valores que no son istorage)<-------
#        return v
#
#    def sync(self):
#        super().sync()
