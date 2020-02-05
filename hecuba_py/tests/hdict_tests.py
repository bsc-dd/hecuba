import unittest

from hecuba import StorageDict


class TestHdict(StorageDict):
    '''
        @TypeSpec dict<<position:int>,text:str>
    '''
    pass


class HdictTest(unittest.TestCase):

    # TEST POSSIBLE WRONG INPUTS

    ######################################################################

    # SAME AS STORAGEOBJ

    ######################################################################

    # IMPLEMENTATION

    # PARSE X DATA

    def test_parse_2(self):
        comment = '''
            @TypeSpec dict<<partid:int>,x:int,y:int,z:int>
            '''
        pd = StorageDict(None,
                         [('pk1', 'int')],
                         [('val1', 'text')])
        p = pd._parse_comments(comment)
        should_be = {
            'columns': [('x', 'int'), ('y', 'int'), ('z', 'int')],
            'primary_keys': [('partid', 'int')],
            'type': 'StorageDict'
        }
        self.assertEquals(p, should_be)
