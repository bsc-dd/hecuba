import unittest

from mock import Mock

from hecuba.IStorage import IStorage
from app.words import Words
from hecuba import config, Config
from hecuba import hdict
from hecuba import StorageDict


class TestHdict(StorageDict):
    '''
        @TypeSpec test dict<<position:int>,text:str>
    '''
    pass


class HdictTest(unittest.TestCase):
    def setUp(self):
        Config.reset(mock_cassandra=True)

    # TEST POSSIBLE WRONG INPUTS

    ######################################################################

    # SAME AS STORAGEOBJ

    ######################################################################

    # IMPLEMENTATION

    # PARSE X DATA

    def test_parse_2(self):
        comment = '''
            @TypeSpec particles dict<<partid:int>,x:int,y:int,z:int>
            '''
        pd = StorageDict(None,
                         [('pk1', 'int')],
                         [('val1', 'text')])
        p = pd._parse_comments(comment)
        should_be = {'particles': {
            'columns': [('x', 'int'), ('y', 'int'), ('z', 'int')],
            'primary_keys': [('partid', 'int')],
            'type': 'StorageDict'
        }}
        self.assertEquals(p, should_be)

    # PARSE X DATA WITH INDEX

    def test_parse_comments(self):
        both2 = {'wordinfo': {'indexed_values': ['wordinfo',
                                                 'position'],
                              'columns': [('wordinfo', 'text')],
                              'primary_keys': [('position',
                                                'int')],
                              'type': 'StorageDict'}}

        both_comment = '''  
                       @TypeSpec wordinfo dict<<position:int>,wordinfo:str>
                       @Index_on wordinfo wordinfo, position
                       '''
        pd = StorageDict(None,
                         [('pk1', 'int')],
                         [('val1', 'text')])
        p = pd._parse_comments(both_comment)
        self.assertEqual(both2, p)

    # PARSE DATA WITH 2 INDEX

    # def test_parse_comments1(self):
    #     pd = StorageDict(None,
    #                      [('pk1', 'int')],
    #                      [('val1', 'text')])
    #     with self.assertRaises(Exception) as context:
    #         input_comment = '''
    #                         @TypeSpec wordinfo dict<<position:int>,wordinfo:str>
    #                         @Index_on wordinfo wordinfo,position
    #                         @Index_on wordinfo wordinfo,position
    #                         '''
    #         pd._parse_comments(input_comment)
    #     self.assertTrue("No valid format", context.exception)

    # PARSE DATA WITH 2 COMMENTS (NO INDEX)

    # def test_parse_comments2(self):
    #     pd = StorageDict(None,
    #                      [('pk1', 'int')],
    #                      [('val1', 'text')])
    #     with self.assertRaises(Exception) as context:
    #         input_comment = '  @TypeSpec wordinfo dict<<position:int>,wordinfo:str>\n ' + \
    #                         '  @TypeSpec a int '
    #         pd._parse_comments(input_comment)
    #     self.assertTrue("No valid format", context.exception)
