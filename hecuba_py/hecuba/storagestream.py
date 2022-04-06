from hecuba import StorageDict
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import Producer, Consumer, TopicPartition
from . import config, log
import pickle

#requirement: only support persistent StorageDicts
class StorageStream(StorageDict):

    def __init__(self, name=None, primary_keys=None, columns=None, indexed_on=None, storage_id=None, **kwargs):
        log.debug("StorageStream: INITIALIZE name {} sid {}".format(name, storage_id))
        if not name and not storage_id:
            raise Exception("Only persistent objects supported")
        super().__init__(name, primary_keys, columns, indexed_on, storage_id, **kwargs)
        self.kafka_names = str.join(",", config.kafka_names)
        self.kafka_consumer = None
        self.topic_name = str(self.storage_id)

        log.debug("Enabling STREAM %s", self.topic_name)
        self._hcache.enable_stream(self.topic_name, {'kafka_names': str.join(",",config.kafka_names)})
        # Producer and Consumer are lazily created only if 'setitem' or 'poll' are invoked.
        self._stream_producer_enabled = False
        self._stream_consumer_enabled = False


# d[42]=666 --> __setitem__

    def __setitem__(self,key, value):
        log.debug("StorageStream: SETITEM sid {} key {} value {}".format(self.storage_id, key, value))
        if not self._stream_producer_enabled:
            self._hcache.enable_stream_producer()
            self._stream_producer_enabled=True
        super().__setitem__(key,value)

    def send(self, key, value):
        log.debug("StorageStream: SEND sid {} key {} value {}".format(self.storage_id, key, value))
        self.__setitem__(key,value)


    def poll(self):
        log.debug("StorageStream: POLL sid {} ".format(self.storage_id))
        import time
        if not self._stream_consumer_enabled:
            self._hcache.enable_stream_consumer()
            self._stream_consumer_enabled=True
        start = time.time()
        row = self._hcache.poll()
        stop = time.time()

        print("poll row type {} got row: {}".format(type(row),row),flush=True)
        v=row[-(len(row)-self._k_size)]
        k=row[0:self._k_size]
        print("poll row type {} got row: {}".format(type(row),row),flush=True)
        print("poll k {} v {}".format(k,v),flush=True)
        #v = super().__getitem__(key)
        self._hcache.add_to_cache(self._make_key(k),self._make_value(v)) #this line only works for StorageDicts
        stop2 = time.time()
        print("TIME POLL {} TIME GETITEM {}".format(stop-start, stop2-stop), flush=True)
        return v

    def sync(self):
        super().sync()
