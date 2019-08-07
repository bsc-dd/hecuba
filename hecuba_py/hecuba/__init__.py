import logging
import os


# Set default log.handler to avoid "No handler found" warnings.

stderrLogger = logging.StreamHandler()
f = '%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s'
stderrLogger.setFormatter(logging.Formatter(f))


log = logging.getLogger('hecuba')
log.addHandler(stderrLogger)

if 'DEBUG' in os.environ and os.environ['DEBUG'].lower() == "true":
    log.setLevel(logging.DEBUG)
elif 'HECUBA_LOG' in os.environ:
    log.setLevel(os.environ['HECUBA_LOG'].upper())
else:
    log.setLevel(logging.ERROR)


from .parser import Parser
from .storageobj import StorageObj
from .hdict import StorageDict
from .hnumpy import StorageNumpy
from .storageiter import StorageIter

__all__ = ['StorageObj', 'StorageDict', 'StorageNumpy', 'Parser', 'StorageIter']