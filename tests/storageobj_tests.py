import unittest

from mock import Mock

from hecuba.IStorage import IStorage
from app.words import Words
from hecuba import config, Config
from hecuba.storageobj import StorageObj


class TestStorageObj(StorageObj):
    '''
       @ClassField test dict<<position:int>,text:str>
    '''
    pass


class StorageObjTest(unittest.TestCase):
    def setUp(self):
        Config.reset(mock_cassandra=True)

    def test_parse_comments(self):
        result = {'instances': {'columns': [('instances',
                                             'counter')],
                                'primary_keys': [('word',
                                                  'text')],
                                'type': 'dict'}}
        result_comment = " @ClassField instances dict<<word:str>,instances:atomicint> "

        p = StorageObj._parse_comments(result_comment)
        self.assertEqual(result, p)

        words = {'wordinfo': {'columns': [('wordinfo', 'text')],
                              'primary_keys': [('position',
                                                'int')],
                              'type': 'dict'}}
        words_comment = '  @ClassField wordinfo dict<<position:int>,wordinfo:str> '
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
        both_comment = '  @ClassField wordinfo dict<<position:int>,wordinfo:str>\n ' + \
                       '@ClassField instances dict<<word:str>,instances:atomicint> '
        p = StorageObj._parse_comments(both_comment)
        self.assertEqual(both, p)

        both2 = {'wordinfo': {'indexed_values': ['wordinfo',
                                                 'position'],
                              'columns': [('wordinfo', 'text')],
                              'primary_keys': [('position',
                                                'int')],
                              'type': 'dict'},
                 'instances': {'indexed_values': ['instances',
                                                  'word'],
                               'columns': [('instances',
                                            'counter')],
                               'primary_keys': [('word',
                                                 'text')],
                               'type': 'dict'
                               }
                 }
        both_comment = '  @ClassField wordinfo dict<<position:int>,wordinfo:str>\n ' + \
                       '  @Index_on instances instances,word\n ' + \
                       '  @ClassField instances dict<<word:str>,instances:atomicint> ' + \
                       '  @Index_on wordinfo wordinfo,position\n '
        p = StorageObj._parse_comments(both_comment)
        self.assertEqual(both2, p)

    def test_parse_2(self):
        comment = "     @ClassField particles dict<<partid:int>,x:int,y:int,z:int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': [('x', 'int'), ('y', 'int'), ('z', 'int')],
            'primary_keys': [('partid', 'int')],
            'type': 'dict'
        }}
        self.assertEquals(p, should_be)

    def test_parse_3(self):
        comment = "     @ClassField particles dict<<partid:int,part2:str>,x:int,y:int,z:int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': [('x', 'int'), ('y', 'int'), ('z', 'int')],
            'primary_keys': [('partid', 'int'), ('part2', 'text')],
            'type': 'dict'
        }}
        self.assertEquals(p, should_be)

    def test_init(self):
        # still in development
        config.session.execute = Mock(return_value=None)
        nopars = Words()
        config.session.execute.assert_not_called()


