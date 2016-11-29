import unittest
from hecuba.globalinit import classfilesparser
from hecuba.settings import *


class GlobalInit_Tests(unittest.TestCase):
    def parsefile_dict_test(self):
        classes = classfilesparser()
        expected_to_be = \
          {'Result':
              {'module_name': 'app.result',
               'storage_objs': {'instances': {'columns': [('instances',
                                                           'counter')],
                                              'primary_keys': [('word',
                                                                'text')],
                                              'type': 'dict'}}},
          'Words': {'module_name': 'app.words',
                    'storage_objs': {'wordinfo': {'columns': [('wordinfo', 'text')],
                                                  'primary_keys': [('position',
                                                                    'int')],
                                                  'type': 'dict'}}}}

        self.assertDictEqual(classes, expected_to_be)
