import datetime
import decimal
import uuid

import numpy
import logging
import os
# User class to Cassandra data type
_hecuba2cassandra_typemap = {
    bool: 'boolean',
    int: 'int',
    float: 'float',
    str: 'text',
    bytearray: 'blob',
    bytes: 'blob',
    tuple: 'tuple',
    frozenset: 'set',
    decimal.Decimal: 'decimal',
    datetime.date: 'date',
    datetime.datetime: 'datetime',
    datetime.time: 'time',
    numpy.int8: 'tinyint',
    numpy.int16: 'smallint',
    numpy.int64: 'double',
    numpy.ndarray: 'hecuba.hnumpy.StorageNumpy',
    uuid.UUID: 'uuid'
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
