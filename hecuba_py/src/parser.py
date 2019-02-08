import re
import ast
from itertools import count
import json

from IStorage import IStorage


class Parser(object):
    args_names = ["type_parser"]

    # @staticmethod
    # def _check_input_types_dict(list_input):
    #     conversions_list = re.sub('[<>():,]', ' ', str(IStorage._conversions))
    #     if not set(list_input[1::2]).issubset(conversions_list.split()):
    #         raise Exception("Incorrect input types introduced")
    #     else:
    #         return
    #
    # @staticmethod
    # def _check_input_types(list_input):
    #     conversions_list = re.sub(r'[^\w]', ' ', str(IStorage._conversions))
    #     if not len(set(list_input) & set(conversions_list.split())):
    #         raise Exception("Incorrect input types introduced")
    #     else:
    #         return
    #
    # @staticmethod
    # def _check_input_index(list_input, new):
    #     if not re.sub(r'[^\w]', ' ', str(list_input[1])) in re.sub(r'[^\w]', ' ',
    #                                                                str(new[list_input[0]]['primary_keys'])):
    #         raise Exception("Incorrect input types introduced")
    #     else:
    #         return
    #
    # @staticmethod
    # def _replace_types(types):
    #     '''Def: replaces the declarated variables with valid Cassandra types.
    #                             Returns: a string with the changed variables.'''
    #     converted = " ".join([IStorage._conversions.get(w, w) for w in types.split()])
    #     return converted
    #
    # @staticmethod
    # def _add_fields(dict_keys, type_field):
    #     '''Def: adds variables to keys and values of the declaration if they doesn't exist.
    #                             Returns: the same declaration with the generated variables in it.'''
    #     split = dict_keys.split(',')
    #     counter = count(0)
    #     result = ', '.join(
    #         [str(type_field + str(counter.next()) + ':' + word.replace(' ', '') + " ") for word in split])
    #     return result
    #
    # @staticmethod
    # def _set_or_tuple_case(param_set, type):
    #     '''Def: constructs a dictionary with the values of the set.
    #                             Returns: a dict structure with inserted values.'''
    #     var = param_set[0]
    #     param_set.pop(0)
    #     aux = {'columns': []}
    #     aux_list = []
    #     if len(param_set) > 1:
    #         counter = count(0)
    #         for type_val in param_set:
    #             aux_list.append((var + '_' + str(counter.next()), type_val))
    #     else:
    #         for type_val in param_set:
    #             aux_list.append((var, type_val))
    #     aux["columns"].append({"primary_keys": aux_list, "type": type})
    #     return aux
    #
    # @staticmethod
    # def _replace_list_types(list_values):
    #     '''Def: replaces the types of list_values to the ones of IStorage (accepted in Cassandra).
    #                             Returns: a list of the converted types'''
    #     final_list = []
    #     for e1, e2 in list_values:
    #         if e2 != 'numpy.ndarray':
    #             var = (e1, IStorage._conversions[e2])
    #             final_list.append(var)
    #     if not final_list:
    #         return list_values
    #     return final_list
    #
    # def _parsing_set(self, line, new):
    #     '''Def: parses set value declaration, checking for the introduced vars.
    #                             Returns: a dict structure with the parsed dict.'''
    #     output = {}
    #     table_name, simple_type = line.groups()
    #     erase_symbols_keys = re.sub('[<>():,]', ' ', simple_type)
    #     erase_symbols_keys = self._replace_types(erase_symbols_keys)
    #     erase_symbols_keys = erase_symbols_keys.split()
    # #    self._check_input_types(erase_symbols_keys)
    #     output["primary_keys"] = {", ".join(erase_symbols_keys)}
    #     output["type"] = 'set'
    #     new[table_name] = output
    #     return new
    #
    # def _parsing_tuple(self, line, new):
    #     '''Def: parses tuple declaration, checking for the introduced vars.
    #                             Returns: a dict structure with the parsed dict.'''
    #     output = {}
    #     table_name, simple_type = line.groups()
    #     erase_symbols_keys = re.sub('[<>():,]', ' ', simple_type)
    #     erase_symbols_keys = self._replace_types(erase_symbols_keys)
    #     erase_symbols_keys = erase_symbols_keys.split()
    # #    self._check_input_types(erase_symbols_keys)
    #     output["columns"] = {", ".join(erase_symbols_keys)}
    #     output["type"] = 'tuple'
    #     new[table_name] = output
    #     return new
    #
    # def _parsing_simple_value(self, line, new):
    #     '''Def: parses simple value declaration, checking for the introduced vars.
    #                             Returns: a dict structure with the parsed dict.'''
    #     output = {}
    #     table_name, simple_type = line.groups()
    #  #   self._check_input_types(simple_type.split())
    #     output["type"] = self._replace_types(simple_type)
    #     new[table_name] = output
    #     return new
    #
    # def _parsing_index(self, line, new):
    #     '''Def: parses index declaration, checking for the introduced vars.
    #                             Returns: a dict structure with the parsed dict.'''
    #     table_name, indexed_values = line.groups()
    #     indexed_values = self._replace_types(indexed_values)
    #     indexed_values = indexed_values.replace(' ', '').split(',')
    #     if table_name in new:
    #     #    self._check_input_index(indexed_values, new)
    #         new[table_name].update({'indexed_values': indexed_values})
    #     else:
    #         new[table_name] = {'indexed_values': indexed_values}
    #     return new
    #
    # def _parsing_keys_and_columns_to_list(self, line):
    #     '''Def: parses keys and values to a list with the dict structure.
    #                             Returns: a list with the dict structure.'''
    #     erase_symbols_keys = re.sub('[<>():,]', ' ', line)
    #     erase_symbols_keys = self._replace_types(erase_symbols_keys)
    #     created_list = erase_symbols_keys.split()
    #  #   self._check_input_types_dict(created_list)
    #     final_list = zip(created_list[::2], created_list[1::2])
    #     return final_list
    #
    # def _check_vars(self, dict_keys, dict_values):
    #     '''Def: checks if the keys and values have variables specified.
    #                             Returns: keys and values with the variables generated.'''
    #     if dict_keys.find(':') == -1:
    #         dict_keys = self._add_fields(dict_keys, "key")
    #         dict_values = self._add_fields(dict_values, "value")
    #     return dict_keys, dict_values
    #
    # def _check_set_in_values(self, dict_values):
    #     '''Def: Checks if there's a set in the dict values.
    #                     Returns: if true, returns the set prepared for the parsing and the splitted set.
    #                              otherwise, returns false.'''
    #     # if dict_values.find('set') != -1:
    #     #     splitted_values = dict_values.split(',')
    #     #     matching = [s for s in splitted_values if 'set' in s]
    #     #     matching_aux = [re.sub('[<>():,]', ' ', f) for f in matching]
    #     #     param_set = [f.replace('set', '') for f in matching_aux]
    #     #     param_set = [f.split() for f in param_set]
    #     #     erase_symbols_keys = self._replace_list_types(param_set)
    #     #     final_set = self._set_case(erase_symbols_keys)
    #     if dict_values.find('set') != -1:
    #         _set_case = getattr(IStorage, self.type_parser + "_set_case_values")
    #         m = _set_case.search(dict_values)
    #         var, parsed_set = m.groups()
    #         matching = str(var)+':set'+'<'+parsed_set+'>'
    #         erase_symbols_keys = re.sub('[<>():,]', ' ', matching)
    #         erase_symbols_keys = erase_symbols_keys.replace('set', '')
    #         erase_symbols_keys = self._replace_types(erase_symbols_keys)
    #         erase_symbols_keys = erase_symbols_keys.split()
    #         final_set = self._set_or_tuple_case(erase_symbols_keys, tuple)
    #         return final_set, matching
    #     return False, False
    #
    # def _check_set_or_tuple_in_values(self, dict_values, type):
    #     '''Def: Checks if there's a set in the dict values.
    #                     Returns: if true, returns the set prepared for the parsing and the splitted set.
    #                              otherwise, returns false.'''
    #     # if dict_values.find('set') != -1:
    #     #     splitted_values = dict_values.split(',')
    #     #     matching = [s for s in splitted_values if 'set' in s]
    #     #     matching_aux = [re.sub('[<>():,]', ' ', f) for f in matching]
    #     #     param_set = [f.replace('set', '') for f in matching_aux]
    #     #     param_set = [f.split() for f in param_set]
    #     #     erase_symbols_keys = self._replace_list_types(param_set)
    #     #     final_set = self._set_case(erase_symbols_keys)
    #     if dict_values.find('set') != -1:
    #         _set_case = getattr(IStorage, self.type_parser + "_set_case_values")
    #         m = _set_case.search(dict_values)
    #         var, parsed_set = m.groups()
    #         matching = str(var)+':set'+'<'+parsed_set+'>'
    #         erase_symbols_keys = re.sub('[<>():,]', ' ', matching)
    #         erase_symbols_keys = erase_symbols_keys.replace('set', '')
    #         erase_symbols_keys = self._replace_types(erase_symbols_keys)
    #         erase_symbols_keys = erase_symbols_keys.split()
    #         final_set = self._set_or_tuple_case(erase_symbols_keys, 'set')
    #         return final_set, matching
    #
    #     elif dict_values.find('tuple') != -1:
    #         _set_case = getattr(IStorage, self.type_parser + "_tuple_case_values")
    #         m = _set_case.search(dict_values)
    #         var, parsed_set = m.groups()
    #         matching = str(var)+':tuple'+'<'+parsed_set+'>'
    #         erase_symbols_keys = re.sub('[<>():,]', ' ', matching)
    #         erase_symbols_keys = erase_symbols_keys.replace('tuple', '')
    #         erase_symbols_keys = self._replace_types(erase_symbols_keys)
    #         erase_symbols_keys = erase_symbols_keys.split()
    #         final_set = self._set_or_tuple_case(erase_symbols_keys, 'tuple')
    #         return final_set, matching
    #
    #     return False, False
    #
    # def _parsing_dict(self, line, new):
    #     '''Def: parses dictionary declaration, checking for the introduced vars.
    #                     Returns: a dict structure with the parsed dict.'''
    #     output = {}
    #     table_name, dict_keys, dict_values = line.groups()
    #     dict_keys, dict_values = self._check_vars(dict_keys, dict_values)
    #     set_val, set_match = self._check_set_or_tuple_in_values(dict_values, 'set')
    #     tuple_val, tuple_match = self._check_set_or_tuple_in_values(dict_values, 'tuple')
    #     if set_match is not False:
    #             dict_values = dict_values.replace(set_match, '')
    #     if tuple_match is not False:
    #             dict_values = dict_values.replace(tuple_match, '')
    #     output["columns"] = self._parsing_keys_and_columns_to_list(dict_values)
    #     output["primary_keys"] = self._parsing_keys_and_columns_to_list(dict_keys)
    #     output["type"] = 'StorageDict'
    #     if set_match is not False:
    #         output["columns"].extend(set_val["columns"])
    #     elif tuple_match is not False:
    #         output["columns"].extend(tuple_val["columns"])
    #     if table_name in new:
    #         new[table_name].update(output)
    #     else:
    #         new[table_name] = output
    #     return new
    #
    # def _parsing_file(self, line, new):
    #     '''Def: Checks if the file declaration is correct.
    #             Returns: the file declaration with a dict structure'''
    #     output = {}
    #     table_name, route = line.groups()
    #     cname, module = IStorage.process_path(route)
    #     try:
    #         mod = __import__(module, globals(), locals(), [cname], 0)
    #     except ValueError:
    #         raise ValueError("Can't import class {} from module {}".format(cname, module))
    #     output["type"] = str(route)
    #     if table_name in new:
    #         new[table_name].update(output)
    #     else:
    #         new[table_name] = output
    #     return new
    #
    # def _fitting_line_type(self, line, this):
    #     '''Def: Gets the splitted line and parses it.
    #             Returns: a dict structure with the parsed line.'''
    #     ret = {}
    #     _dict_case = getattr(IStorage, self.type_parser + "_dict_case")
    #     _tuple_case = getattr(IStorage, self.type_parser + "_tuple_case")
    #     _simple_case = getattr(IStorage, self.type_parser + "_simple_case")
    #     _set_case = getattr(IStorage, self.type_parser + "_set_case")
    #     _file_case = getattr(IStorage, self.type_parser + "_file_case")
    #     _dict_tuple_key_case = getattr(IStorage, self.type_parser + "_tuple_case_key")
    #     _index_case = getattr(IStorage, "_index_vars")
    #
    #     if _dict_case.match(line) is not None:
    #         # Matching dict
    #         ret = self._parsing_dict(_dict_case.match(line), this)
    #
    #     elif _dict_tuple_key_case(line) is not None:
    #         # Matching keys dict (tuple or set)
    #         ret = self._parsing_dict(_dict_tuple_key_case(line), this)
    #
    #     elif _index_case.match(line) is not None:
    #         # Matching Index_on
    #         ret = self._parsing_index(_index_case.match(line), this)
    #     elif _tuple_case.match(line) is not None:
    #         # Matching tuple
    #         ret = self._parsing_tuple(_tuple_case.match(line), this)
    #     elif _set_case.match(line) is not None:
    #         # Matching set
    #         ret = self._parsing_set(_set_case.match(line), this)
    #     elif _simple_case.match(line) is not None:
    #         # Matching simple type
    #         ret = self._parsing_simple_value(_simple_case.match(line), this)
    #     elif _file_case.match(line) is not None and line.find('numpy') != 0:
    #         # Matching file
    #         ret = self._parsing_file(_file_case.match(line), this)
    #         # Not matching
    #     if this == {}:
    #         raise Exception("Incorrect input types introduced")
    #     this = ret
    #     return this
    #
    # def _comprovation_input_elements(self, comments):
    #     '''Def: Checks if the comments introduced of the type TypeSpec are in the correct format.
    #             and checks if the comments introduced are not duplicated.
    #             Returns: false if there's some wrong comment specification, true otherwise.'''
    #     list_coms = comments.split('\n')
    #     #CHECKS NOT IMPLEMENTED
    #     return True


    # def _parse_comments(self, comments):
    #     '''Def: Parses the comments param to a ClassField or TypeSpec type and checks if the comments are in the correct
    #             format.
    #             Returns: an structure with all the parsed comments.'''
    #     this = {}
    #     '''Erasing first and last line'''
    #     str_splitted = comments.split('\n', 1)[-1]
    #     lines = str_splitted.rsplit('\n', 1)[0]
    #     ''''''
    #     if self.type_parser == "TypeSpec":
    #        # self._comprovation_input_elements(comments)
    #         for line in lines.split('\n'):
    #             this = self._fitting_line_type(self._remove_spaces_from_line(line), this)
    #     elif self.type_parser == "ClassField":
    #         for line in lines.split('\n'):
    #             this = self._fitting_line_type(self._remove_spaces_from_line(line), this)
    #     return this

    #######################################################
    # def _get_keys_values(self, varsk, varsv, cleank, cleanv, typek, typevv):
    #     concatenated_keys = ""
    #     values = ""
    #     string_str = ""
    #     for t, t1, t2 in zip(cleank, varsk, typek):  # first keys
    #         if t2 == 'set':
    #             t = t.split(',')
    #             converted_primary_keys = ", ".join([IStorage._conversions.get(w, w) for w in t])
    #             converted_primary_keys = converted_primary_keys.split(',')
    #             converted_primary_keys = [w.replace(' ', '') for w in converted_primary_keys]
    #             aux_list = []  # stores ((var_1, val),(var_2, val),...)
    #             if(len(converted_primary_keys) > 1):
    #                 counter = count(0)
    #                 for type_val in converted_primary_keys:
    #                     aux_list.append((t1 + '_' + str(counter.next()), type_val))
    #                     string_str = ',{"type": "set", "primary_keys": %s}' % aux_list
    #             else:
    #                 aux_list.append((t1, converted_primary_keys[0]))
    #                 string_str = ',{"type": "set", "primary_keys": %s}' % aux_list
    #
    #         elif t2 == 'tuple':
    #             t = t.split(',')
    #             converted_primary_keys = ", ".join([IStorage._conversions.get(w, w) for w in t])
    #             converted_primary_keys = converted_primary_keys.split(',')
    #             converted_primary_keys = [w.replace(' ', '') for w in converted_primary_keys]
    #             aux_list = []  # stores ((var_1, val),(var_2, val),...)
    #             if (len(converted_primary_keys) > 1):
    #                 counter = count(0)
    #                 for type_val in converted_primary_keys:
    #                     aux_list.append((t1 + '_' + str(counter.next()), type_val))
    #                     string_str = ',{"type": "tuple", "columns": %s}' % aux_list
    #             else:
    #                 aux_list.append((t1, converted_primary_keys[0]))
    #                 string_str = ',{"type": "tuple", "columns": %s}' % aux_list
    #         else:
    #             type = IStorage._conversions[t]
    #             string_str = ',("%s", "%s")' % (t1, type)
    #         concatenated_keys = concatenated_keys + string_str
    #
    #     for t, t1, t2 in zip(cleanv, varsv, typevv):  # first keys
    #         if t2 == 'set':
    #             t = t.split(',')
    #             converted_primary_keys = ", ".join([IStorage._conversions.get(w, w) for w in t])
    #             converted_primary_keys = converted_primary_keys.split(',')
    #             converted_primary_keys = [w.replace(' ', '') for w in converted_primary_keys]
    #             aux_list = []  # stores ((var_1, val),(var_2, val),...)
    #             if (len(converted_primary_keys) > 1):
    #                 counter = count(0)
    #                 for type_val in converted_primary_keys:
    #                     aux_list.append((t1 + '_' + str(counter.next()), type_val))
    #                     string_str = ',{"type": "set", "primary_keys": %s}' % aux_list
    #             else:
    #                 aux_list.append((t1, converted_primary_keys[0]))
    #                 string_str = ',{"type": "set", "primary_keys": %s}' % aux_list
    #
    #         elif t2 == 'tuple':
    #             t = t.split(',')
    #             converted_primary_keys = ", ".join([IStorage._conversions.get(w, w) for w in t])
    #             converted_primary_keys = converted_primary_keys.split(',')
    #             converted_primary_keys = [w.replace(' ', '') for w in converted_primary_keys]
    #             aux_list = []  # stores ((var_1, val),(var_2, val),...)
    #             if (len(converted_primary_keys) > 1):
    #                 counter = count(0)
    #                 for type_val in converted_primary_keys:
    #                     aux_list.append((t1 + '_' + str(counter.next()), type_val))
    #                     string_str = ',{"type": "tuple", "columns": %s}' % aux_list
    #             else:
    #                 aux_list.append((t1, converted_primary_keys[0]))
    #                 string_str = ',{"type": "tuple", "columns": %s}' % aux_list
    #         else:
    #             type = IStorage._conversions[t]
    #             string_str = ',("%s", "%s")' % (t1, type)
    #         values = values + string_str
    #
    #     concatenated_keys = concatenated_keys[1:]
    #     values = values[1:]
    #
    #     return concatenated_keys, values

    #def _clean_vars(self, vars):
    #    return [v.replace(':', '') for v in vars]


    def _get_str_primary_keys_values(self, pk):
        pk = pk[5:]
        count = 0
        pos = 0
        for c in pk:
            pos = pos + 1
            if c == '<':
                count = count + 1
            elif c == '>':
                count = count - 1
            if count == 0: break
        keys = pk[1:pos - 1]
        values = pk[pos + 1:len(pk) - 1]

        # We get the variables

        varsk = re.findall(r"\w+:", keys)  # var keys
        varsv = re.findall(r"\w+:", values)  # var values

        # Now we clean the variables

        varskc = [v.replace(':', '') for v in varsk]
        varsvc = [v.replace(':', '') for v in varsv]

        # We get the valuesk

        for var in varsk:
            keys = keys.replace(var, ' ')

        valsc = keys[1:].split(', ')  # all valuesk separated by comma

        typevk = []
        finalvarsk = []
        for var in valsc:
            aux = var
            if var.count("tuple") > 0:
                typevk.append("tuple")
                aux = aux.replace('tuple', '').replace('<', '').replace('>', '')
            elif var.count("set") > 0:
                typevk.append("set")
                aux = aux.replace('set', '').replace('<', '').replace('>', '')
            else:
                typevk.append("simple")
            finalvarsk.append(aux)

        # We get the valuesv

        for var in varsv:
            values = values.replace(var, ' ')

        valsc1 = values[1:].split(', ')  # all valuesk separated by comma

        typevv = []
        finalvarsv = []
        for var in valsc1:
            aux = var
            if var.count("tuple") > 0:
                typevv.append("tuple")
                aux = aux.replace('tuple', '').replace('<', '').replace('>', '')
            elif var.count("set") > 0:
                typevv.append("set")
                aux = aux.replace('set', '').replace('<', '').replace('>', '')
            else:
                typevv.append("simple")
            finalvarsv.append(aux)

        return varskc, varsvc, finalvarsk, finalvarsv, typevk, typevv

    def _set_or_tuple(self, type, pk_col, t, t1):
        string_str = ""
        t = t.split(',')
        converted_primary_keys = ", ".join([IStorage._conversions.get(w, w) for w in t])
        converted_primary_keys = converted_primary_keys.split(',')
        converted_primary_keys = [w.replace(' ', '') for w in converted_primary_keys]
        aux_list = []  # stores ((var_1, val),(var_2, val),...)
        if (len(converted_primary_keys) > 1):
            counter = count(0)
            for type_val in converted_primary_keys:
                aux_list.append((t1 + '_' + str(counter.next()), type_val))
                string_str = ',{"type": "%s", "%s": %s}' % (type, pk_col, aux_list)
        else:
            aux_list.append((t1, converted_primary_keys[0]))
            string_str = ',{"type": "%s", "%s": %s}' % (type, pk_col, aux_list)
        return string_str

    def _get_dict_str(self, varsk, cleank, typek):
        concatenated_keys = ""
        values = ""
        string_str = ""
        for t, t1, t2 in zip(cleank, varsk, typek):  # first keys
            if t2 == 'set':
                string_str = self._set_or_tuple('set', 'primary_keys', t, t1)
            elif t2 == 'tuple':
                string_str = self._set_or_tuple('tuple', 'columns', t, t1)
            else:
                type = IStorage._conversions[t]
                string_str = ',("%s", "%s")' % (t1, type)
            concatenated_keys = concatenated_keys + string_str
        concatenated_keys = concatenated_keys[1:]
        return concatenated_keys

    def _parse_dict(self, line, this):
        split_line = line.split()
        pk = split_line[2]
        table = split_line[1]
        varsk, varsv, cleank, cleanv, typek, typevv = self._get_str_primary_keys_values(pk)
        pks = self._get_dict_str(varsk, cleank, typek)
        values = self._get_dict_str(varsv, cleanv, typevv)
        final_dict = '{"%s": {"primary_keys": [%s], "columns": [%s], "type": "StorageDict"}}' % (table, pks, values)
        final_dict = eval(final_dict)
        aux = '{"primary_keys": [%s], "columns": [%s], "type": "StorageDict"}' % (pks, values)
        if (table in this):
            this[table].update(eval(aux))
            return this
        return final_dict

    def _parse_set_or_tuple(self, type, line, pk_or_col, this):
        split_line = line.split()
        table = split_line[1]
        line = re.sub('[<>, ]', ' ', split_line[2].replace(str(type), ""))
        primary_keys = line.split()
        converted_primary_keys = ", ".join([IStorage._conversions.get(w, w) for w in primary_keys])
        string_str = '{"%s":{"%s": {"%s"},"type": "%s"}}' % (table, pk_or_col, converted_primary_keys, str(type))
        final_string = eval(string_str)
        aux = '{"%s": {"%s"},"type": "%s"}' % (pk_or_col, converted_primary_keys, str(type))
        if (table in this):
            this[table].update(eval(aux))
            return this
        return final_string

    def _parse_index(self, line, this):
        '''Def: parses index declaration, checking for the introduced vars.
                                Returns: a dict structure with the parsed dict.'''

        table = line.split()[1]
        atributes = line.split(' ', 2)
        atributes = atributes[2].replace(" ", '')
        atributes = atributes.split(',')
        converted_atributes = ", ".join([IStorage._conversions.get(w, w) for w in atributes])
        converted_atributes = converted_atributes.split(',')
        converted_atributes = [w.replace(" ", "") for w in converted_atributes]
        if table in this:
            this[table].update({'indexed_values': converted_atributes})
        else:
            this[table] = {'indexed_values': converted_atributes}
        return this

    def _parse_file(self, line, new):
        '''Def: Checks if the file declaration is correct.
                Returns: the file declaration with a dict structure'''
        line = line.split(" ")
        output = {}
        table_name = line[1]
        route = line[2]
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

    def _parse_set_tuple(self, line, this):
        if(line.count('set')) > 0:
            return self._parse_set_or_tuple('set', line, 'primary_keys', this)
        elif(line.count('tuple')) > 0:
            return self._parse_set_or_tuple('tuple', line, 'columns', this)


    def _parse_simple(self, line, this):
        split_line = line.split()
        table = split_line[1]
        type = IStorage._conversions[split_line[2]]
        simple = '{"%s":{"type":"%s"}}' % (table, type)
        simple = eval(simple)
        if(table in this):
            this[table].update(simple)
        return simple

    def _input_type(self, line, this):
        if (line.count('<') == 1):  # is tuple, set, list
            aux = (self._parse_set_tuple(line, this))
        elif (line.count('<') == 0 and line.count('Index_on') == 0 and line.count('.') == 0 or line.count('numpy.ndarray')):  # is simple type
            aux = (self._parse_simple(line, this))
        elif (line.count('Index_on') == 1):
            aux = self._parse_index(line, this)
        elif (line.count('.') > 0):
            aux = self._parse_file(line, this)
        else:  # is dict
            aux = (self._parse_dict(line, this))
        return aux

    def _remove_spaces_from_line(self, line):
        '''Def: Remove all the spaces of the line splitted from comments
                Returns: same line with no spaces.'''
        line = re.sub(' +', '*', line)
        if (line.count('tuple') == 1 and line.count('dict') == 0):
            pos = line.find('tuple')
        elif (line.count('set') == 1 and line.count('dict') == 0):
            pos = line.find('set')
        elif(line.count('@Index_on') == 1):
            pos = line.find('@Index_on')
            line = line[pos:]
            return line.replace('*', ' ')
        else:
            pos = line.find('dict')

        line = line[0:pos].replace('*', ' ') + line[pos:].replace("*", '')
        return line

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
            for line in lines.split('\n'):
                this = self._input_type(self._remove_spaces_from_line(line), this)
        if self.type_parser == "ClassField":
            for line in lines.split('\n'):
                this.update(self._input_type(self._remove_spaces_from_line(line), this))
        return this

    def __init__(self, type_parser):
        '''Initializes the Parser class with the type_parser that can be @ClassField or @TypeSpec.'''
        self.type_parser = type_parser
