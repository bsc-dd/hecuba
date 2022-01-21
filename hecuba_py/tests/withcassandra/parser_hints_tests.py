import unittest

from hecuba import StorageObj, StorageDict


# no dict
class Dict1(StorageDict):
    '''
    @TypeSpec <<a:int>, b:int, c:str>
    '''


# missing comma
class Dict2(StorageDict):
    '''
    @TypeSpec dict<<a:int>, b:int c:int>
    '''


# bad characters
class Dict3(StorageDict):
    '''
    @TypeSpec dict<<a:int>, b:int, c:int!?¿>
    '''


# two lines
class Dict4(StorageDict):
    '''
    @TypeSpec dict<<a:int>, b:int, c:int>
    @TypeSpec dict<<a:int>, b:int, c:int>
    '''


# no <<key>, value> format
class Dict5(StorageDict):
    '''
    @TypeSpec dict <a:int, b:int>
    '''

# key without name
class Dict6(StorageDict):
    '''
    @TypeSpec dict<<:int>, b:int>
    '''

# key without type
class Dict7(StorageDict):
    '''
    @TypeSpec dict <<key:>, b:int>
    '''

# bad character :
class Obj1(StorageObj):
    '''
    @ClassField a int
    @ClassField b:int
    '''


# possible character :
class Obj2(StorageObj):
    '''
    @ClassField a int
    @ClassField dictField dict<<key0:int>, val0:str>
    '''


# two values in one ClassField
class Obj3(StorageObj):
    '''
    @ClassField a int, b str
    '''


def try_parser(constructor):
    try:
        constructor()
    except Exception as ex:
        return str(ex)
    return None


class ParserHintsTest(unittest.TestCase):

    def test_no_dict(self):
        error = \
            "No detected keys. Maybe you forgot to set a primary key or there is a missing 'dict' after the TypeSpec."
        self.assertEqual(try_parser(Dict1), error)

    def test_missing_coma_dict(self):
        error = "Error parsing Type Specification. Trying to parse: 'b:intc:int'"
        self.assertEqual(try_parser(Dict2), error)

    def test_bad_characters_dict(self):
        error = "One or more bad character detected: [!, ?, ¿]."
        self.assertEqual(try_parser(Dict3), error)

    def test_two_lines_dict(self):
        error = "StorageDicts should only have one TypeSpec line."
        self.assertEqual(try_parser(Dict4), error)

    def test_bad_format_dict(self):
        error = "The TypeSpec should have at least two '<' and two '>'. Format: @TypeSpec dict<<key:type>, value:type>."
        self.assertEqual(try_parser(Dict5), error)

    def test_missing_name_attr(self):
        error = "Error parsing Type Specification. Trying to parse: ':int'"
        self.assertEqual(try_parser(Dict6), error)

    def test_missing_type_attr(self):
        error = "Error parsing Type Specification. Trying to parse: 'key:'"
        self.assertEqual(try_parser(Dict7), error)

    def test_bad_colon_obj(self):
        error = "The ClassField @ClassField b:int should only have the character ':' if it is in a dict."
        self.assertEqual(try_parser(Obj1), error)

    def test_good_colon_obj(self):
        error = None
        self.assertEqual(try_parser(Obj2), error)

    def test_two_values_one_classfield_obj(self):
        error = "Type 'int,' not identified."
        self.assertEqual(try_parser(Obj3), error)


if __name__ == "__main__":
    unittest.main()
