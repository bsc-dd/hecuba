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

    def test_parse_comments(self):
        result = {'instances': {'columns': [('instances',
                                             'counter')],
                                'primary_keys': [('word',
                                                  'text')],
                                'type': 'StorageDict'}}
        result_comment = " @ClassField instances dict<<word:str>,instances:atomicint> "

        p = StorageObj._parse_comments(result_comment)
        self.assertEqual(result, p)

        words = {'wordinfo': {'columns': [('wordinfo', 'text')],
                              'primary_keys': [('position',
                                                'int')],
                              'type': 'StorageDict'}}
        words_comment = '  @ClassField wordinfo dict<<position:int>,wordinfo:str> '
        p = StorageObj._parse_comments(words_comment)
        self.assertEqual(words, p)

        both = {'wordinfo': {'columns': [('wordinfo', 'text')],
                             'primary_keys': [('position',
                                               'int')],
                             'type': 'StorageDict'},
                'instances': {'columns': [('instances',
                                           'counter')],
                              'primary_keys': [('word',
                                                'text')],
                              'type': 'StorageDict'}
                }
        both_comment = "'''\n@ClassField wordinfo dict<<position:int>,wordinfo:str>\n " + \
                       "@ClassField instances dict<<word:str>,instances:atomicint>\n''' "
        p = StorageObj._parse_comments(both_comment)
        self.assertEqual(both, p)

        both2 = {'wordinfo': {'columns': [('wordinfo', 'text')],
                              'primary_keys': [('position',
                                                'int')],
                              'type': 'StorageDict'},
                 'instances': {'columns': [('instances',
                                            'counter')],
                               'primary_keys': [('word',
                                                 'text')],
                               'type': 'StorageDict'
                               }
                 }
        both_comment = "'''\n@ClassField wordinfo dict<<position:int>,wordinfo:str>\n " + \
                       "  @ClassField instances dict<<word:str>,instances:atomicint>\n'''"
        p = StorageObj._parse_comments(both_comment)
        self.assertEqual(both2, p)

    def test_parse_2(self):
        comment = "     @ClassField particles dict<<partid:int>,x:int,y:int,z:int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': [('x', 'int'), ('y', 'int'), ('z', 'int')],
            'primary_keys': [('partid', 'int')],
            'type': 'StorageDict'
        }}
        self.assertEquals(p, should_be)

    def test_parse_3(self):
        comment = "     @ClassField particles dict<<partid:int,part2:str>,x:int,y:int,z:int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': [('x', 'int'), ('y', 'int'), ('z', 'int')],
            'primary_keys': [('partid', 'int'), ('part2', 'text')],
            'type': 'StorageDict'
        }}
        self.assertEquals(p, should_be)

    def test_init(self):
        # still in development
        config.session.execute = Mock(return_value=None)
        nopars = Words()
        config.session.execute.assert_not_called()

    def test_init_pdict(self):
        t = TestStorageObj()
        t.test = {1: 'ciao'}
        #its not persistent, so in memory it is still a dict
        #hecuba converts the dicts to StorageDicts when the StorageObj is made persistent
        self.assertTrue(issubclass(t.test.__class__, dict))


