from . import log
from .IStorage import IStorage

class StorageStream(IStorage):
    def __init__(self, *args, **kwargs):
        #log.debug("StorageStream: __init__")
        print("StorageStream: __init__ {}".format(kwargs), flush=True)
        self._stream_enabled = True
        super().__init__(*args, **kwargs)
        print("StorageStream: _stream_enabled {}".format(self._stream_enabled), flush=True)
