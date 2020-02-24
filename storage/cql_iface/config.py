import datetime
import decimal
import logging
import os
import uuid
from typing import Tuple

import numpy

# User class to Cassandra data type
from storage.cql_iface.tests.mockStorageObj import StorageObj

_hecuba2cassandra_typemap = {
    bool: 'boolean',
    int: 'int',
    float: 'float',
    str: 'text',
    bytearray: 'blob',
    bytes: 'blob',
    Tuple: 'tuple',
    tuple: 'tuple',
    # FrozenSet: 'set',
    decimal.Decimal: 'decimal',
    datetime.date: 'date',
    datetime.datetime: 'timestamp',
    datetime.time: 'time',
    numpy.int8: 'tinyint',
    numpy.int16: 'smallint',
    numpy.int64: 'double',
    numpy.ndarray: 'uuid',
    uuid.UUID: 'uuid',
    StorageObj: 'uuid'
}

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
