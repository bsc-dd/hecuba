import unittest

from mock import Mock

from hecuba.IStorage import IStorage
from app.words import Words
from hecuba import config, Config
from hecuba import StorageObj


class TestStorageObj(StorageObj):
    '''
       @ClassField test dict<<position:int>,text:str>
    '''
    pass


class StorageObjTest(unittest.TestCase):
    def setUp(self):
        Config.reset(mock_cassandra=True)

    # TEST POSSIBLE WRONG INPUTS

    ######################################################################

    # BLANK INPUTS

    ######################################################################

    # ALL INPUTS BLANK
    def test_blank_input(self):
        with self.assertRaises(Exception) as context:
            input_comment = " "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    # NO DATA INPUT
    def test_no_data_introduced(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_data_on_index(self):
        with self.assertRaises(Exception) as context:
            input_comment = " @Index_on wordinfo "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_table_parser_introduced_simple_type_classfield(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField int"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_table_parser_introduced_dictionary_type_classfield(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField dict<<key:str>,value:atomicint>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_table_parser_introduced_tuple_type_classfield(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField tuple <int, str, bool> "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_table_parser_introduced_set_type_classfield(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField set<int> "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    # NO TABLE, NO DATA INPUT (@CLASSFIELD)
    def test_no_table_no_data_input_classfield(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    # NO @TYPESPEC / @CLASSFIELD
    def test_no_ts_cs_simple_type(self):
        with self.assertRaises(Exception) as context:
            input_comment = "table int"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_ts_cs_dictionary_type(self):
        with self.assertRaises(Exception) as context:
            input_comment = "table dict<<key:str>,value:atomicint>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_ts_cs_tuple_type(self):
        with self.assertRaises(Exception) as context:
            input_comment = "table tuple <int, str, bool> "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_ts_cs_set_type(self):
        with self.assertRaises(Exception) as context:
            input_comment = "table set<int> "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    # NO @TYPESPEC / @CLASSFIELD, NO INPUT DATA
    def test_no_ts_cs_data(self):
        with self.assertRaises(Exception) as context:
            input_comment = "table"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    # NO @TYPESPEC / @CLASSFIELD, NO TABLE
    def test_no_ts_cs_table_simple_type(self):
        with self.assertRaises(Exception) as context:
            input_comment = "int"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_ts_cs_table_dict_type(self):
        with self.assertRaises(Exception) as context:
            input_comment = "dict<<key:str>,atomicint>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_ts_cs_table_tuple_type(self):
        with self.assertRaises(Exception) as context:
            input_comment = "tuple <int, str, bool> "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_no_ts_cs_table_set_type(self):
        with self.assertRaises(Exception) as context:
            input_comment = "set<int> "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    ######################################################################

    # TYPO INPUT ERRORS

    ######################################################################

    # INCORRECT FORMAT INTRODUCED

    def test_wrong_simple_type_format_introduced_classfield_1(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table <<int>>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_simple_type_format_introduced_classfield_2(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table <int>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_simple_type_format_introduced_classfield_3(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table <<int>,>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_dictionary_format_introduced_classfield_1(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table dict<key:str>,val:atomicint>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_dictionary_format_introduced_classfield_2(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table dict<key:str>,>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_dictionary_format_introduced_classfield_3(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table dict<key:str>,val:int>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_tuple_format_introduced_classfield_1(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table tuple int, str, bool> "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_set_format_introduced_classfield_1(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table set<int "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    # VARIABLES TYPE ERROR

    def test_wrong_type_parser_introduced_simple_type_classfield(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table inta"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_type_parser_introduced_dictionary_type_classfield_1(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table dicta<<key:stra>,value:atomicinta>"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_type_parser_introduced_tuple_type_classfield_1(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table tupla <int, str, bool> "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_type_parser_introduced_set_type_classfield_1(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField table seta<int> "
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    def test_wrong_file_parsing(self):
        with self.assertRaises(Exception) as context:
            input_comment = "@ClassField myfile a.b.c"
            StorageObj._parse_comments(input_comment)
        self.assertTrue("Incorrect input types introduced", context.exception)

    ######################################################################

    # IMPLEMENTED METHODS

    ######################################################################

    # GENERAL CASE (NO SET DICTS)

    def test_parse_comments(self):
        result = {'instances': {'columns': [('instances',
                                             'counter')],
                                'primary_keys': [('word',
                                                  'text')],
                                'type': 'StorageDict'}}
        result_comment = '''
                         @ClassField instances dict<<word:str>, instances:atomicint> 
                         '''

        p = StorageObj._parse_comments(result_comment)
        self.assertEqual(result, p)

        words = {'wordinfo': {'columns': [('wordinfo', 'text')],
                              'primary_keys': [('position',
                                                'int')],
                              'type': 'StorageDict'}}
        words_comment = '''
                        @ClassField wordinfo dict<<position:int>, wordinfo:str> 
                        '''
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
        both_comment = ''' 
                            @ClassField wordinfo dict<<position:int>, wordinfo:str>
                            @ClassField instances dict<<word:str>, instances:atomicint> 
                       '''
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
        both_comment = '''  
                        @ClassField wordinfo dict<<position:int>, wordinfo:str>
                        @Index_on instances instances, word
                        @ClassField instances dict<<word:str>, instances:atomicint>
                        @Index_on wordinfo wordinfo, position 
                       '''
        p = StorageObj._parse_comments(both_comment)
        self.assertEqual(both2, p)

    # DICT CASE

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
        comment = "     @ClassField particles dict<<partid:int, part2:str>,x:int,y:int, z:int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': [('x', 'int'), ('y', 'int'), ('z', 'int')],
            'primary_keys': [('partid', 'int'), ('part2', 'text')],
            'type': 'StorageDict'
        }}
        self.assertEquals(p, should_be)

    def test_parse_4(self):
        comment = "     @ClassField particles dict<<int,str>, int,int,int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': [('value0', 'int'), ('value1', 'int'), ('value2', 'int')],
            'primary_keys': [('key0', 'int'), ('key1', 'text')],
            'type': 'StorageDict'
        }}
        self.assertEquals(p, should_be)

    # SIMPLE VALUE CASE

    def test_parse_5(self):
        comment = "     @ClassField table bool"
        p = StorageObj._parse_comments(comment)
        should_be = {'table': {
            'type': 'boolean'
        }}
        self.assertEquals(p, should_be)

    # TUPLE CASE

    def test_parse_8(self):
        comment = "     @ClassField particles tuple<int, str, bool>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': {'int, text, boolean'},
            'type': 'tuple'
        }}
        self.assertEquals(p, should_be)

    def test_parse_6(self):
        comment = "     @ClassField particles tuple<int, int, int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles': {
            'columns': {'int, int, int'},
            'type': 'tuple'
        }}
        self.assertEquals(p, should_be)

    # SET CASE

    def test_parse_7(self):
        comment = "     @ClassField particles2 set<int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles2': {
            'primary_keys': {'int'},
            'type': 'set'
        }}
        self.assertEquals(p, should_be)

    def test_parse_7(self):
        comment = "     @ClassField particles2 set<int,bool,int>"
        p = StorageObj._parse_comments(comment)
        should_be = {'particles2': {
            'primary_keys': {'int, boolean, int'},
            'type': 'set'
        }}
        self.assertEquals(p, should_be)

    # TEST COMPILATION
    def test_index(self):
        comment = '''
                  @ClassField test dict<<pos:int>,word:str>
                  @Index_on test word
                  '''
        p = StorageObj._parse_comments(comment)
        should_be = {'test': {'indexed_values': ['word'],
                              'columns': [('word', 'text')],
                              'primary_keys': [('pos',
                                                'int')],
                              'type': 'StorageDict'}
                     }
        self.assertEquals(p, should_be)

    def test_parse_all(self):
        comment = '''  
                  @ClassField wordinfo dict<<position:int>,wordinfo:str>
                  @Index_on instances instances, word
                  @ClassField instances dict<<word:str>,instances:atomicint>
                  @Index_on wordinfo wordinfo, position
                  @ClassField table int
                  @ClassField particles tuple<int, str, bool>
                  @ClassField particles2 set<int>
                  '''
        p = StorageObj._parse_comments(comment)
        should_be = {'wordinfo': {'indexed_values': ['wordinfo',
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
                                   },
                     'table': {
                         'type': 'int'
                     },
                     'particles': {
                         'columns': {'int, text, boolean'},
                         'type': 'tuple'
                     },
                     'particles2': {
                         'primary_keys': {'int'},
                         'type': 'set'
                     }
                     }
        self.assertEquals(p, should_be)

    # SET DICT

    def test_dict_set(self):
        comment = '  @ClassField wordinfo dict<<position:int>, atr2:int, atr:int, dif:set<int>>'
        p = StorageObj._parse_comments(comment)
        should_be = {'wordinfo': {'primary_keys': [('position', 'int')],
                                  'columns': [('atr2', 'int'), ('atr', 'int'),
                                              {'type': 'set', 'primary_keys': [('dif', 'int')], }],
                                  'type': 'StorageDict'}}
        self.assertEquals(p, should_be)

    # SET DICT (NO VARS)

    def test_dict_set_no_name(self):
        comment = '  @ClassField wordinfo dict<<int>, int, int, set<int>>'
        p = StorageObj._parse_comments(comment)
        should_be = {'wordinfo': {'primary_keys': [('key0', 'int')],
                                  'columns': [('value0', 'int'), ('value1', 'int'),
                                              {'type': 'set', 'primary_keys': [('value2', 'int')], }],
                                  'type': 'StorageDict'}}
        self.assertEquals(p, should_be)

    def test_dict_set_name(self):
        comment = '  @ClassField wordinfo dict<<key0:int>, value0:int, value1:int, value2:set<int, int>>'
        p = StorageObj._parse_comments(comment)
        should_be = {'wordinfo': {'primary_keys': [('key0', 'int')],
                                  'columns': [('value0', 'int'), ('value1', 'int'),
                                              {'type': 'set', 'primary_keys': [('value2_0', 'int'), ('value2_1', 'int')]}],
                                  'type': 'StorageDict'}}
        self.assertEquals(p, should_be)

    # MULTI SET DICT

    # def test_dict_multi_set(self):
    #     comment = '  @ClassField wordinfo dict<<position:int>,atr2:int, atr:int,dif:set<int>, dif2:set<bool>>'
    #     p = StorageObj._parse_comments(comment)
    #     should_be = {'wordinfo': {'primary_keys': [('position', 'int')],
    #                               'columns': [('atr2', 'int'), ('atr', 'int'),
    #                                           {'type': 'set', 'primary_keys': [('dif', 'int')]},
    #                                           {'type': 'set', 'primary_keys': [('dif2', 'boolean')]}],
    #                               'type': 'StorageDict'}}
    #     self.assertEquals(p, should_be)
    #
    # # MULTI SET DICT (NO VARS)
    #
    # def test_dict_multi_set_no_name(self):
    #     comment = '  @ClassField wordinfo dict<<int>, int, int, set<int>, set<bool>>'
    #     p = StorageObj._parse_comments(comment)
    #     should_be = {'wordinfo': {'primary_keys': [('key0', 'int')],
    #                               'columns': [('value0', 'int'), ('value1', 'int'),
    #                                           {'type': 'set', 'primary_keys': [('value2', 'int')]},
    #                                           {'type': 'set', 'primary_keys': [('value3', 'boolean')]}],
    #                               'type': 'StorageDict'}}
    #     self.assertEquals(p, should_be)

    def test_file_parsing(self):
        comment = "@ClassField myfile tests.app.words.Words"
        p = StorageObj._parse_comments(comment)
        should_be = {'myfile': {'type': 'tests.app.words.Words'}}
        self.assertEquals(p, should_be)

    # def test_dict_multi_set_no_name(self):
    #     comment = '''
    #     @ClassField wordinfo dict<<int>, int, numpy.ndarray, set<numpy.ndarray>, set<numpy.ndarray>>
    #     '''
    #     p = StorageObj._parse_comments(comment)
    #     should_be = {'wordinfo': {'primary_keys': [('key0', 'int')],
    #                               'columns': [('value0', 'int'), ('value1', 'numpy.ndarray'),
    #                                           {'type': 'set', 'primary_keys': [('value2', 'numpy.ndarray')]},
    #                                           {'type': 'set', 'primary_keys': [('value3', 'numpy.ndarray')]}],
    #                               'type': 'StorageDict'}}
    #     self.assertEquals(p, should_be)

    def test_dict_(self):
        comment = '''
        @ClassField words dict<<position:int>, wordinfo:str>
        '''
        p = StorageObj._parse_comments(comment)
        should_be = {'words': {'primary_keys': [('position', 'int')],
                               'columns': [('wordinfo', 'text')],
                               'type': 'StorageDict'}}
        self.assertEquals(p, should_be)

    def test_numpy_(self):
        comment = '''
        @ClassField words numpy.ndarray
        '''
        p = StorageObj._parse_comments(comment)
        should_be = {'words': {
            'type': 'numpy.ndarray'
        }}
        self.assertEquals(p, should_be)


    ##########################################################################################

    # TO BE TESTED
