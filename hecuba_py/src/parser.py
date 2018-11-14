import re
from itertools import count

from IStorage import IStorage


class Parser(object):
    args_names = ["type_parser", "comments"]

    @staticmethod
    def _check_input_types_dict(list_input):
        conversions_list = re.sub(r'[^\w]', ' ', str(IStorage._conversions))
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
        converted = " ".join([IStorage._conversions.get(w, w) for w in types.split()])
        return converted

    @staticmethod
    def _add_fields(dict_keys, type_field):
        split = dict_keys.split(',')
        counter = count(0)
        result = ', '.join(
            [str(type_field + str(counter.next()) + ':' + word.replace(' ', '') + " ") for word in split])
        return result

    @staticmethod
    def _set_case(param_set):
        aux = {}
        aux["columns"] = []
        for var, type in param_set:
            aux["columns"].append({"primary_keys": [(var, type)], "type": 'set'})
        return aux

    @staticmethod
    def _replace_list_types(list):
        final_list = []
        for e1, e2 in list:
            var = (e1, IStorage._conversions[e2])
            final_list.append(var)
        return final_list

    def _parsing_set(self, line, new):
        output = {}
        _set_case = getattr(IStorage, self.type_parser + "_set_case")
        m = _set_case.match(line)
        table_name, simple_type = m.groups()
        erase_symbols_keys = re.sub(r'[^\w]', ' ', simple_type)
        repl_types = self._replace_types(erase_symbols_keys)
        self._check_input_types(repl_types.split())
        output["primary_keys"] = {repl_types}
        output["type"] = 'set'
        new[table_name] = output
        return new

    def _parsing_tuple(self, line, new):
        output = {}
        _tuple_case = getattr(IStorage, self.type_parser + "_tuple_case")
        m = _tuple_case.match(line)
        table_name, simple_type = m.groups()
        erase_symbols_keys = re.sub(r'[^\w]', ' ', simple_type)
        erase_symbols_keys = self._replace_types(erase_symbols_keys)
        erase_symbols_keys = erase_symbols_keys.split()
        self._check_input_types(erase_symbols_keys)
        output["columns"] = {", ".join(erase_symbols_keys)}
        output["type"] = 'tuple'
        new[table_name] = output
        return new

    def _parsing_simple_value(self, line, new):
        output = {}
        _simple_case = getattr(IStorage, self.type_parser + "_simple_case")
        _simple_case = _simple_case.match(line)
        table_name, simple_type = _simple_case.groups()
        self._check_input_types(simple_type.split())
        output["type"] = self._replace_types(simple_type)
        new[table_name] = output
        return new

    def _parsing_index(self, line, new):
        # _index_vars = re.compile('.*@Index_on *([\w]+) + *([\w]+)+(, \w+)*')
        _index_vars = re.compile('.*@Index_on *([A-z0-9]+) +([A-z0-9, ]+)')
        match = _index_vars.match(line)
        table_name, indexed_values = match.groups()
        indexed_values = self._replace_types(indexed_values)
        indexed_values = indexed_values.replace(' ', '').split(',')
        if table_name in new:
            self._check_input_index(indexed_values, new)
            new[table_name].update({'indexed_values': indexed_values})
        else:
            new[table_name] = {'indexed_values': indexed_values}
        return new

    def _parsing_keys_and_columns_to_list(self, line):
        erase_symbols_keys = re.sub(r'[^\w]', ' ', line)
        erase_symbols_keys = self._replace_types(erase_symbols_keys)
        created_list = erase_symbols_keys.split()
        self._check_input_types_dict(created_list)
        final_list = zip(created_list[::2], created_list[1::2])
        return final_list

    def _check_vars(self, dict_keys, dict_values):
        if dict_keys.find(':') == -1:
            dict_keys = self._add_fields(dict_keys, "key")
            dict_values = self._add_fields(dict_values, "value")
        return dict_keys, dict_values

    def _check_set_in_values(self, dict_values):
        if dict_values.find('set') != -1:
            splited_values = dict_values.split(',')
            matching = [s for s in splited_values if 'set' in s]
            matching_aux = [re.sub(r'[^\w]', ' ', f) for f in matching]
            param_set = [f.replace('set', '') for f in matching_aux]
            param_set = [f.split() for f in param_set]
            erase_symbols_keys = self._replace_list_types(param_set)
            final_set = self._set_case(erase_symbols_keys)
            return final_set, matching
        return False, False

    def _parsing_dict(self, line, new):
        output = {}
        _dict_case = getattr(IStorage, self.type_parser + "_dict_case")
        match_dict = _dict_case.match(line)
        table_name, dict_keys, dict_values = match_dict.groups()
        dict_keys, dict_values = self._check_vars(dict_keys, dict_values)
        dic, set = self._check_set_in_values(dict_values)
        if set is not False:
            for s in set:
                dict_values = dict_values.replace(s, '')
        output["columns"] = self._parsing_keys_and_columns_to_list(dict_values)
        output["primary_keys"] = self._parsing_keys_and_columns_to_list(dict_keys)
        output["type"] = 'StorageDict'
        if dic is not False:
            output["columns"].extend(dic["columns"])
        if table_name in new:
            new[table_name].update(output)
        else:
            new[table_name] = output
        return new

    def _fitting_line_type(self, line, this):
        ret = {}
        _dict_case = getattr(IStorage, self.type_parser + "_dict_case")
        _tuple_case = getattr(IStorage, self.type_parser + "_tuple_case")
        _simple_case = getattr(IStorage, self.type_parser + "_simple_case")
        _set_case = getattr(IStorage, self.type_parser + "_set_case")
        _index_check = IStorage._index_case.match(line)
        if _index_check is not None:
            table, table2, values = _index_check.groups()
        else:
            table = table2 = values = None

        if _dict_case.match(line) is not None:
            # Matching @ClassField of a dict
            ret = self._parsing_dict(line, this)
        elif table and table2 and values is not None and table == table2:
            # Matching @Index_on
            ret = self._parsing_index(line, this)
        elif _tuple_case.match(line) is not None:
            # Matching @ClassField of a tuple
            ret = self._parsing_tuple(line, this)
        elif _set_case.match(line) is not None:
            # Matching @ClassField of a set
            ret = self._parsing_set(line, this)
        elif _simple_case.match(line) is not None:
            # Matching simple type
            ret = self._parsing_simple_value(line, this)
        if this == {}:
            raise Exception("Incorrect input types introduced")
        this = ret
        return this

    def _comprovation_input_elements(self, comments):
        list_coms = comments.split('\n')
        # list_coms = [e.replace(' ', '') for e in list_coms]
        if ((not len(list_coms)) == 1) or (((len(list_coms)) > 1) and comments.find("@Index_on") == -1):
            raise Exception('No valid format')
        if len(list_coms) != len(set(list_coms)):
            raise Exception('Duplicated comments')
        # POSSIBLE INDEX_ON COMPROBATIONS
        return True

    def _parse_comments(self, comments):
        # self._repeated_comments(comments)
        this = {}
        if self.type_parser == "TypeSpec":
            self._comprovation_input_elements(comments)
            for line in comments.split('\n'):
                this = self._fitting_line_type(line, this)
        elif self.type_parser == "ClassField":
            for line in comments.split('\n'):
                this = self._fitting_line_type(line, this)
        return this

    def __init__(self, type_parser):
        self.type_parser = type_parser
