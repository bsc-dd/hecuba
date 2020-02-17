import unittest
import uuid
from datetime import time, date, datetime
from typing import Tuple

import numpy
from hecuba import StorageObj


class TestInt(StorageObj):
    attr1: int


class TestBool(StorageObj):
    attr1: bool


class TestFloat(StorageObj):
    attr1: float


class TestStr(StorageObj):
    attr1: str


class TestBytearray(StorageObj):
    attr1: bytearray


class TestBytes(StorageObj):
    attr1: bytes


class TestTuple(StorageObj):
    attr1: Tuple[int, int]


class TestTime(StorageObj):
    attr1: time


class TestDate(StorageObj):
    attr1: date


class TestDatetime(StorageObj):
    attr1: datetime


class TestTinyint(StorageObj):
    attr1: numpy.int8


class TestSmallint(StorageObj):
    attr1: numpy.int16


class TestDouble(StorageObj):
    attr1: numpy.int64


class TestUuid(StorageObj):
    attr1: uuid.UUID


class TestNumpy(StorageObj):
    attr1: numpy.ndarray


class MyTestCase(unittest.TestCase):
    # def test_numpy(self):  # TODO: NEEDS TO BE IMPLEMENTED
    #     a = TestUuid("test_numpy")
    #     a.attr1 = numpy.arange(10)
    #     a = TestUuid("test_numpy")
    #     self.assertEqual(a.attr1, [uuid.UUID("123e4567-e89b-12d3-a456-426655440000")])

    def test_uuid(self):
        a = TestUuid("test_uuid")
        a.attr1 = uuid.UUID("123e4567-e89b-12d3-a456-426655440000")
        a = TestUuid("test_uuid")
        self.assertEqual(a.attr1, [uuid.UUID("123e4567-e89b-12d3-a456-426655440000")])

    def test_int(self):
        a = TestInt("test_int")
        a.attr1 = 1000000
        a = TestInt("test_int")
        self.assertEqual(a.attr1, [1000000])

    def test_bool(self):
        a = TestBool("test_bool")
        a.attr1 = True
        a = TestBool("test_bool")
        self.assertEqual(a.attr1, [True])

    def test_float(self):
        a = TestFloat("test_float")
        a.attr1 = 3.14
        a = TestFloat("test_float")
        self.assertAlmostEqual(a.attr1[0], 3.14, places=3)

    def test_string(self):
        a = TestStr("test_string")
        a.attr1 = "hola"
        a = TestStr("test_string")
        self.assertEqual(a.attr1, ["hola"])

    # def test_bytearray(self):
    #     a = TestBytearray("test_bytearray")
    #     a.attr1 = bytearray(b'\x00')
    #     a = TestBytearray("test_bytearray")
    #     self.assertEqual(a.attr1, [bytearray(b'\x00')])

    # def test_bytes(self):
    #     a = TestBytes("test_bytes")
    #     a.attr1 = bytes("a", 'utf-8')
    #     a = TestBytes("test_bytes")
    #     self.assertEqual(a.attr1, ['\x01'])

    def test_tuple(self):
        a = TestTuple("test_tuple")
        a.attr1 = (1, 2)
        a = TestTuple("test_tuple")
        self.assertEqual(a.attr1, [(1, 2)])

    def test_time(self):
        a = TestTime("test_time")
        a.attr1 = time(12, 10, 30)
        a = TestTime("test_time")
        self.assertEqual(a.attr1, [time(12, 10, 30)])

    def test_date(self):
        a = TestDate("test_date")
        a.attr1 = date(2020, 2, 1)
        a = TestDate("test_date")
        self.assertEqual(a.attr1, [date(2020, 2, 1)])

    def test_datetime(self):
        a = TestDatetime("test_datetime")
        a.attr1 = datetime(2020, 2, 1, 12, 10, 30)
        a = TestDatetime("test_datetime")
        self.assertEqual(a.attr1, [datetime(2020, 2, 1, 12, 10, 30)])

    # def test_tinyint(self):
    #     a = TestTinyint("test_tinyint")
    #     a.attr1 = numpy.int8(5)
    #     a = TestTinyint("test_tinyint")
    #     self.assertEqual(a.attr1, [numpy.int8(5)])
    #
    # def test_smallint(self):
    #     a = TestSmallint("test_smallint")
    #     a.attr1 = numpy.int16(32000)
    #     a = TestSmallint("test_smallint")
    #     self.assertEqual(a.attr1, [numpy.int16(32000)])
    #
    # def test_double(self):
    #     a = TestDouble("test_double")
    #     a.attr1 = numpy.int64(50000000000)
    #     a = TestDouble("test_double")
    #     self.assertEqual(a.attr1, [numpy.int64(50000000000)])

if __name__ == '__main__':
    unittest.main()
