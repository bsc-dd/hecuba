import re
from itertools import count

from IStorage import IStorage


class Parser(object):
    args_names = ["type_parser"]

    @staticmethod
    def _check_input_types_dict(list_input):
        conversions_list = re.sub('[<>():,]', ' ', str(IStorage._conversions))
        if not set(list_input[1::2]).issubset(conversions_list.split()):
            raise Exception("Incorrect input types introduced")
        else:
            return

    @staticmethod
    def _check_input_types(list_input):
        conversions_list = re.sub(r'[^\w]', ' ', str(IStorage._conversions))
        if not len(set(list_input) & set(conversions_list.split())):
            raise Exception("Incorrect input types introduced")
        else:
            return

    @staticmethod
    def _check_input_index(list_input, new):
        if not re.sub(r'[^\w]', ' ', str(list_input[1])) in re.sub(r'[^\w]', ' ',
                                                                   str(new[list_input[0]]['primary_keys'])):
            raise Exception("Incorrect input types introduced")
        else:
            return

    @staticmethod
    def _replace_types(types):
        '''Def: replaces the declarated variables with valid Cassandra types.
                                Returns: a string with the changed variables.'''
        converted = " ".join([IStorage._conversions.get(w, w) for w in types.split()])
        return converted

    @staticmethod
    def _add_fields(dict_keys, type_field):
        '''Def: adds variables to keys and values of the declaration if they doesn't exist.
                                Returns: the same declaration with the generated variables in it.'''
        split = dict_keys.split(',')
        counter = count(0)
        result = ', '.join(
            [str(type_field + str(counter.next()) + ':' + word.replace(' ', '') + " ") for word in split])
        return result

    @staticmethod
    def _set_case(param_set):
        '''Def: constructs a dictionary with the values of the set.
                                Returns: a dict structure with inserted values.'''
        var = param_set[0]
        param_set.pop(0)
        aux = {'columns': []}
        aux_list = []
        if len(param_set) > 1:
            counter = count(0)
            for type_val in param_set:
                aux_list.append((var + '_' + str(counter.next()), type_val))
        else:
            for type_val in param_set:
                aux_list.append((var, type_val))
        aux["columns"].append({"primary_keys": aux_list, "type": 'set'})
        return aux

    @staticmethod
    def _replace_list_types(list_values):
        '''Def: replaces the types of list_values to the ones of IStorage (accepted in Cassandra).
                                Returns: a list of the converted types'''
        final_list = []
        for e1, e2 in list_values:
            if e2 != 'numpy.ndarray':
                var = (e1, IStorage._conversions[e2])
                final_list.append(var)
        if not final_list:
            return list_values
        return final_list

    def _parsing_set(self, line, new):
        '''Def: parses set value declaration, checking for the introduced vars.
                                Returns: a dict structure with the parsed dict.'''
        output = {}
        table_name, simple_type = line.groups()
        erase_symbols_keys = re.sub('[<>():,]', ' ', simple_type)
        erase_symbols_keys = self._replace_types(erase_symbols_keys)
        erase_symbols_keys = erase_symbols_keys.split()
    #    self._check_input_types(erase_symbols_keys)
        output["primary_keys"] = {", ".join(erase_symbols_keys)}
        output["type"] = 'set'
        new[table_name] = output
        return new

    def _parsing_tuple(self, line, new):
        '''Def: parses tuple declaration, checking for the introduced vars.
                                Returns: a dict structure with the parsed dict.'''
        output = {}
        table_name, simple_type = line.groups()
        erase_symbols_keys = re.sub('[<>():,]', ' ', simple_type)
        erase_symbols_keys = self._replace_types(erase_symbols_keys)
        erase_symbols_keys = erase_symbols_keys.split()
    #    self._check_input_types(erase_symbols_keys)
        output["columns"] = {", ".join(erase_symbols_keys)}
        output["type"] = 'tuple'
        new[table_name] = output
        return new

    def _parsing_simple_value(self, line, new):
        '''Def: parses simple value declaration, checking for the introduced vars.
                                Returns: a dict structure with the parsed dict.'''
        output = {}
        table_name, simple_type = line.groups()
     #   self._check_input_types(simple_type.split())
        output["type"] = self._replace_types(simple_type)
        new[table_name] = output
        return new

    def _parsing_index(self, line, new):
        '''Def: parses index declaration, checking for the introduced vars.
                                Returns: a dict structure with the parsed dict.'''
        table_name, indexed_values = line.groups()
        indexed_values = self._replace_types(indexed_values)
        indexed_values = indexed_values.replace(' ', '').split(',')
        if table_name in new:
        #    self._check_input_index(indexed_values, new)
            new[table_name].update({'indexed_values': indexed_values})
        else:
            new[table_name] = {'indexed_values': indexed_values}
        return new

    def _parsing_keys_and_columns_to_list(self, line):
        '''Def: parses keys and values to a list with the dict structure.
                                Returns: a list with the dict structure.'''
        erase_symbols_keys = re.sub('[<>():,]', ' ', line)
        erase_symbols_keys = self._replace_types(erase_symbols_keys)
        created_list = erase_symbols_keys.split()
     #   self._check_input_types_dict(created_list)
        final_list = zip(created_list[::2], created_list[1::2])
        return final_list

    def _check_vars(self, dict_keys, dict_values):
        '''Def: checks if the keys and values have variables specified.
                                Returns: keys and values with the variables generated.'''
        if dict_keys.find(':') == -1:
            dict_keys = self._add_fields(dict_keys, "key")
            dict_values = self._add_fields(dict_values, "value")
        return dict_keys, dict_values

    def _check_set_in_values(self, dict_values):
        '''Def: Checks if there's a set in the dict values.
                        Returns: if true, returns the set prepared for the parsing and the splitted set.
                                 otherwise, returns false.'''
        # if dict_values.find('set') != -1:
        #     splitted_values = dict_values.split(',')
        #     matching = [s for s in splitted_values if 'set' in s]
        #     matching_aux = [re.sub('[<>():,]', ' ', f) for f in matching]
        #     param_set = [f.replace('set', '') for f in matching_aux]
        #     param_set = [f.split() for f in param_set]
        #     erase_symbols_keys = self._replace_list_types(param_set)
        #     final_set = self._set_case(erase_symbols_keys)
        if dict_values.find('set') != -1:
            _set_case = getattr(IStorage, self.type_parser + "_set_case_values")
            m = _set_case.search(dict_values)
            var, parsed_set = m.groups()
            matching = str(var)+':set'+'<'+parsed_set+'>'
            erase_symbols_keys = re.sub('[<>():,]', ' ', matching)
            erase_symbols_keys = erase_symbols_keys.replace('set', '')
            erase_symbols_keys = self._replace_types(erase_symbols_keys)
            erase_symbols_keys = erase_symbols_keys.split()
            final_set = self._set_case(erase_symbols_keys)
            return final_set, matching
        return False, False

    def _parsing_dict(self, line, new):
        '''Def: parses dictionary declaration, checking for the introduced vars.
                        Returns: a dict structure with the parsed dict.'''
        output = {}
        table_name, dict_keys, dict_values = line.groups()
        dict_keys, dict_values = self._check_vars(dict_keys, dict_values)
        set_val, set_match = self._check_set_in_values(dict_values)
        if set_match is not False:
                dict_values = dict_values.replace(set_match, '')
        output["columns"] = self._parsing_keys_and_columns_to_list(dict_values)
        output["primary_keys"] = self._parsing_keys_and_columns_to_list(dict_keys)
        output["type"] = 'StorageDict'
        if set_match is not False:
            output["columns"].extend(set_val["columns"])
        if table_name in new:
            new[table_name].update(output)
        else:
            new[table_name] = output
        return new

    def _parsing_file(self, line, new):
        '''Def: Checks if the file declaration is correct.
                Returns: the file declaration with a dict structure'''
        output = {}
        table_name, route = line.groups()
        cname, module = IStorage.process_path(route)
        try:
            mod = __import__(module, globals(), locals(), [cname], 0)
        except ValueError:
            raise ValueError("Can't import class {} from module {}".format(cname, module))
        output["type"] = str(route)
        if table_name in new:
            new[table_name].update(output)
        else:
            new[table_name] = output
        return new

    def _fitting_line_type(self, line, this):
        '''Def: Gets the splitted line and parses it.
                Returns: a dict structure with the parsed line.'''
        ret = {}
        _dict_case = getattr(IStorage, self.type_parser + "_dict_case")
        _tuple_case = getattr(IStorage, self.type_parser + "_tuple_case")
        _simple_case = getattr(IStorage, self.type_parser + "_simple_case")
        _set_case = getattr(IStorage, self.type_parser + "_set_case")
        _file_case = getattr(IStorage, self.type_parser + "_file_case")
        _index_case = getattr(IStorage, "_index_vars")
        if _dict_case.match(line) is not None:
            # Matching dict
            ret = self._parsing_dict(_dict_case.match(line), this)
        elif _index_case.match(line) is not None:
            # Matching Index_on
            ret = self._parsing_index(_index_case.match(line), this)
        elif _tuple_case.match(line) is not None:
            # Matching tuple
            ret = self._parsing_tuple(_tuple_case.match(line), this)
        elif _set_case.match(line) is not None:
            # Matching set
            ret = self._parsing_set(_set_case.match(line), this)
        elif _simple_case.match(line) is not None:
            # Matching simple type
            ret = self._parsing_simple_value(_simple_case.match(line), this)
        elif _file_case.match(line) is not None and line.find('numpy') != 0:
            # Matching file
            ret = self._parsing_file(_file_case.match(line), this)
            # Not matching
        if this == {}:
            raise Exception("Incorrect input types introduced")
        this = ret
        return this

    def _comprovation_input_elements(self, comments):
        '''Def: Checks if the comments introduced of the type TypeSpec are in the correct format.
                and checks if the comments introduced are not duplicated.
                Returns: false if there's some wrong comment specification, true otherwise.'''
        list_coms = comments.split('\n')
        #CHECKS NOT IMPLEMENTED
        return True

    def _remove_spaces_from_line(self, line):
        '''Def: Remove all the spaces of the line splitted from comments
                Returns: same line with no spaces.'''
        nospace = re.sub(' +', '*', line)
        return nospace.replace('*', ' ')

    def _parse_comments(self, comments):
        '''Def: Parses the comments param to a ClassField or TypeSpec type and checks if the comments are in the correct
                format.
                Returns: an structure with all the parsed comments.'''
        this = {}
        '''Erasing first and last line'''
        str_splitted = comments.split('\n', 1)[-1]
        lines = str_splitted.rsplit('\n', 1)[0]
        ''''''
        if self.type_parser == "TypeSpec":
           # self._comprovation_input_elements(comments)
            for line in lines.split('\n'):
                this = self._fitting_line_type(self._remove_spaces_from_line(line), this)
        elif self.type_parser == "ClassField":
            for line in lines.split('\n'):
                this = self._fitting_line_type(self._remove_spaces_from_line(line), this)
        return this

    def __init__(self, type_parser):
        '''Initializes the Parser class with the type_parser that can be @ClassField or @TypeSpec.'''
        self.type_parser = type_parser
