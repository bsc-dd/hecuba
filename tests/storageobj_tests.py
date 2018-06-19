import unittest

from mock import Mock


class StorageObjTest(unittest.TestCase):
    import hecuba
    Config = hecuba.Config

    def setUp(self):
        self.Config.reset(mock_cassandra=True)

    def test_parse_comments(self):
        from hecuba.storageobj import StorageObj
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
        both_comment = '  @ClassField wordinfo dict<<position:int>,wordinfo:str>\n ' + \
                       '@ClassField instances dict<<word:str>,instances:atomicint> '
        p = StorageObj._parse_comments(both_comment)
        self.assertEqual(both, p)

        both2 = {'wordinfo': {'indexed_values': ['wordinfo',
                                                 'position'],
                              'columns': [('wordinfo', 'text')],
                              'primary_keys': [('position',
                                                'int')],
                              'type': 'StorageDict'},
                 'instances': {'indexed_values': ['instances',
                                                  'word'],
                               'columns': [('instances',
                                            'counter')],
                               'primary_keys': [('word',
                                                 'text')],
                               'type': 'StorageDict'
                               }
                 }
        both_comment = '  @ClassField wordinfo dict<<position:int>,wordinfo:str>\n ' + \
                       '  @Index_on instances instances,word\n ' + \
                       '  @ClassField instances dict<<word:str>,instances:atomicint> ' + \
                       '  @Index_on wordinfo wordinfo,position\n '
        p = StorageObj._parse_comments(both_comment)
        self.assertEqual(both2, p)

    def test_parse_2(self):
        from hecuba.storageobj import StorageObj
        comment = "     @ClassField particles dict<<partid:int>,x:int,y:int,z:int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': [('x', 'int'), ('y', 'int'), ('z', 'int')],
            'primary_keys': [('partid', 'int')],
            'type': 'StorageDict'
        }}
        self.assertEquals(p, should_be)

    def test_parse_3(self):
        from hecuba.storageobj import StorageObj
        comment = "     @ClassField particles dict<<partid:int,part2:str>,x:int,y:int,z:int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': [('x', 'int'), ('y', 'int'), ('z', 'int')],
            'primary_keys': [('partid', 'int'), ('part2', 'text')],
            'type': 'StorageDict'
        }}
        self.assertEquals(p, should_be)

    def test_init(self):
        from hecuba import config
        from class_definitions import Words
        # still in development
        config.session.execute = Mock(return_value=None)
        nopars = Words()
        config.session.execute.assert_not_called()

    def test_init_pdict(self):
        from class_definitions import TestStorageObj
        t = TestStorageObj()
        t.test = {1: 'ciao'}
        #its not persistent, so in memory it is still a dict
        #hecuba converts the dicts to StorageDicts when the StorageObj is made persistent
        self.assertTrue(issubclass(t.test.__class__, dict))


