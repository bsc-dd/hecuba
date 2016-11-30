import unittest

from mock import Mock

from hecuba.storageobj import StorageObj


class StorageObjTest(unittest.TestCase):
    def parse_comments_test(self):
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
        both_comment = '  @ClassField wordinfo dict <<position:int>,wordinfo:str>\n '+\
                        '@ClassField instances dict <<word:str>,instances:atomicint> '
        p = StorageObj._parse_comments(both_comment)
        self.assertEqual(both, p)

