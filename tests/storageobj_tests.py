import unittest

from hecuba.dict import PersistentDict
from mock import Mock, call

from hecuba import config

from hecuba.storageobj import StorageObj


class StorageObjTest(unittest.TestCase):

    def setUp(self):
        config.reset(mock_cassandra=True)

    def test_parse_comments(self):
        result = {'instances': {'columns': [('instances',
                                             'counter')],
                                'primary_keys': [('word',
                                                  'text')],
                                'type': 'dict'}}
        result_comment = " @ClassField instances dict <<word:str>,instances:atomicint> "

        p = StorageObj._parse_comments(result_comment)
        self.assertEqual(result, p)

        words = {'wordinfo': {'columns': [('wordinfo', 'text')],
                              'primary_keys': [('position',
                                                'int')],
                              'type': 'dict'}}
        words_comment = '  @ClassField wordinfo dict <<position:int>,wordinfo:str> '
        p = StorageObj._parse_comments(words_comment)
        self.assertEqual(words, p)

        both = {'wordinfo': {'columns': [('wordinfo', 'text')],
                             'primary_keys': [('position',
                                               'int')],
                             'type': 'dict'},
                'instances': {'columns': [('instances',
                                           'counter')],
                              'primary_keys': [('word',
                                                'text')],
                              'type': 'dict'}
                }
        both_comment = '  @ClassField wordinfo dict <<position:int>,wordinfo:str>\n ' + \
                       '@ClassField instances dict <<word:str>,instances:atomicint> '
        p = StorageObj._parse_comments(both_comment)
        self.assertEqual(both, p)



    def test_init(self):
        config.session.execute = Mock(return_value=None)
        nopars = StorageObj('ksp1.tt1', myuuid='ciao')
        self.assertEqual('tt1', nopars._table)
        self.assertEqual('ksp1', nopars._ksp)
        self.assertEqual('ciao', nopars._myuuid)
        config.session.execute.assert_not_called()


    def test_build_remotely(self):
        config.session.execute = Mock(return_value=None)
        class res:pass
        r =res()
        r.ksp = 'ksp1'
        r.tab = 'tt1'
        r.blockid = 'ciao'
        r.storageobj_classname = "hecuba.storageobj.StorageObj"
        nopars = StorageObj.build_remotely(r)
        self.assertEqual('tt1', nopars._table)
        self.assertEqual('ksp1', nopars._ksp)
        self.assertEqual('ciao', nopars._myuuid)
        config.session.execute.assert_not_called()


    def test_init_create_pdict(self):
        config.session.execute = Mock(return_value=None)
        from app.result import Result
        nopars = Result('ksp1.tt1', myuuid='ciao')
        self.assertEqual('tt1', nopars._table)
        self.assertEqual('ksp1', nopars._ksp)
        self.assertEqual('ciao', nopars._myuuid)
        self.assertEqual(True, nopars._persistent)
        self.assertTrue(hasattr(nopars, 'instances'))
        self.assertIsInstance(nopars.instances, PersistentDict)
        config.session.execute.assert_not_called()

    def test__set_attr(self):
        config.session.execute = Mock(return_value=None)
        nopars = StorageObj('ksp1.tt1', myuuid='ciao')
        nopars.ciao = 1
        config.session.execute.assert_called_with('INSERT INTO ksp1.tt1(name,intval) VALUES (%s,%s)', ['ciao', 1])

        config.session.execute = Mock(return_value=None)
        nopars = StorageObj('ksp1.tt1', myuuid='ciao')
        nopars.ciao = "1"
        config.session.execute.assert_called_with('INSERT INTO ksp1.tt1(name,textval) VALUES (%s,%s)', ['ciao', "1"])

        config.session.execute = Mock(return_value=None)
        nopars = StorageObj('ksp1.tt1', myuuid='ciao')
        nopars.ciao = [1, 2, 3]
        config.session.execute.assert_called_with('INSERT INTO ksp1.tt1(name,intlist) VALUES (%s,%s)', ['ciao', [1,2,3]])

        config.session.execute = Mock(return_value=None)
        nopars = StorageObj('ksp1.tt1', myuuid='ciao')
        nopars.ciao = (1, 2, 3)
        config.session.execute.assert_called_with('INSERT INTO ksp1.tt1(name,inttuple) VALUES (%s,%s)',
                                           ['ciao', [1, 2, 3]])

    def test_set_and_get(self):
        config.session.execute = Mock(return_value=[])
        nopars = StorageObj('ksp1.tb1', myuuid='ciao')
        self.assertTrue(nopars._persistent)
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        try:   nopars.ciao;
        except KeyError:pass
        try:   nopars.ciao2;
        except KeyError:pass
        try:   nopars.ciao3;
        except KeyError:pass
        try:   nopars.ciao4;
        except KeyError:pass


        calls = [call('INSERT INTO ksp1.tb1(name,intval) VALUES (%s,%s)', ['ciao', 1]),
                 call('INSERT INTO ksp1.tb1(name,textval) VALUES (%s,%s)', ['ciao2', "1"]),
                 call('INSERT INTO ksp1.tb1(name,intlist) VALUES (%s,%s)', ['ciao3', [1, 2, 3]]),
                 call('INSERT INTO ksp1.tb1(name,inttuple) VALUES (%s,%s)', ['ciao4', [1, 2, 3]]),

                 call('SELECT intval FROM ksp1.tb1 WHERE name = %s', ['ciao']),
                 call('SELECT textval FROM ksp1.tb1 WHERE name = %s', ['ciao2']),
                 call('SELECT intlist FROM ksp1.tb1 WHERE name = %s', ['ciao3']),
                 call('SELECT inttuple FROM ksp1.tb1 WHERE name = %s', ['ciao4'])
                 ]

        config.session.execute.assert_has_calls(calls, any_order=True)

