import re
from itertools import count

from .tools import process_path

_conversions = {'atomicint': 'counter',
                'str': 'text',
                'bool': 'boolean',
                'decimal': 'decimal',
                'float': 'float',
                'int': 'int',
                'tuple': 'tuple',
                'list': 'list',
                'generator': 'list',
                'frozenset': 'set',
                'set': 'set',
                'dict': 'map',
                'long': 'bigint',
                'buffer': 'blob',
                'bytearray': 'blob',
                'counter': 'counter',
                'double': 'double',
                'StorageDict': 'dict',
                'ndarray': 'hecuba.hnumpy.StorageNumpy',
                'numpy.ndarray': 'hecuba.hnumpy.StorageNumpy',
                'date': 'date',
                'time': 'time',
                'datetime': 'timestamp'}


class Parser(object):
    args_names = ["type_parser"]
    split_dtypes_regex = re.compile('^(tuple|set)<(.*)>$')

    def _append_values_to_list_after_replace(self, vals):
        """
        Receives a list of data types. Strips the outermost data type.
        Returns:
            typev: list of the outer data types, with the keyword "simple" if not found
            finalvars: list of the corresponding internal data types
        """
        typev = []
        finalvars = []
        for var in vals:
            res = self.split_dtypes_regex.search(var)
            if res:
                typev.append(res.group(1))
                finalvars.append(res.group(2))
            else:
                typev.append("simple")
                finalvars.append(var)
        return typev, finalvars

    def _get_elements(self, s):
        """
        Args:
            s is a string with a type specification containing one or more types
        Returns  a list of type_specifications
        Example:
            k1:tuple<int,int>,k2:tuple<int,str>,str
            -->
            'k1:tuple<int,int>' 'k2:tuple<int,str>' 'str'
        """
        elements=[]
        n_brackets = 0
        pos = 0
        lastpos = 0
        for pos, c in enumerate(s):
            if c == '<':
                n_brackets = n_brackets + 1
            elif c == '>':
                n_brackets = n_brackets - 1
            elif c == ',':
                if n_brackets == 0: # a new element found
                    elements.append( s[lastpos:pos] )
                    lastpos = pos + 1 # skip ','
        if lastpos < pos: #add last element
            elements.append( s[lastpos:] )
        return elements

    def _get_name_and_type(self, k):
        """
        Args:
            k is a string with a single type specification "name:value"
        Return:
            name and type, or None and type if ":" is not present
            Raises Syntax Error in cases: "n:", ":v" , ":"
        """
        s = k.split(":")
        if len(s) == 2: # case "name:value"
            if len(s[0]) > 0 and len(s[1]) > 0:
                return s
        elif len(s) == 1: # case "value" only
            if len(s[0]) > 0: # case ":"
                return None, s[0]
        raise SyntaxError("Error parsing Type Specification. Trying to parse: '{}'".format(k))

    def _get_str_primary_keys_values(self, pk):
        """
        Args:
            pk is a string with a dict specification "dict<<key_specification>, value_specification>"
        Return:
            Six lists:
                - keys' names,
                - values' names,
                - keys' types (simple, tuple or set),
                - values' types (simple, tuple or set),
                - keys' types (int, float, ...),
                - values' types (int, float, ...),
        """
        pk = pk.replace("dict", "", 1).strip()

        # Find point to split keys from values
        n_brackets = 0
        pos = 0
        for pos, c in enumerate(pk):
            if c == '<':
                n_brackets = n_brackets + 1
            elif c == '>':
                n_brackets = n_brackets - 1
                if n_brackets == 1:
                    break

        keys = pk[2:pos]
        values = pk[pos + 2:len(pk) - 1]

        if not keys:
            raise SyntaxError("Can't detect the keys in the TypeSpec")

        # We get the variables
        keyList = self._get_elements(keys)
        valueList = self._get_elements(values)
        # Parse Keys...
        keyNamesList = []
        keyTypesList = []
        for i,k in enumerate(keyList):
            myname,mytype = self._get_name_and_type(k)
            if not myname: # Generate name "key_0","key_1",...,"key_N"
                myname = "key_" + str(i)
            keyNamesList.append(myname)
            keyTypesList.append(mytype)

        # Parse Values...
        valueNamesList = []
        valueTypesList = []
        offset = len(keyNamesList)
        for i,v in enumerate(valueList):
            myname,mytype = self._get_name_and_type(v)
            if not myname: # Generate name "val_N+1","valN+2",...
                myname = "val_" + str(i + offset)
            valueNamesList.append(myname)
            valueTypesList.append(mytype)


        # for each type we store if its a 'simple' or a 'tuple/set' type
        # (finalvarksk and finalvarsv)
        # and for 'set' or 'tuple' types, the type specification is replaced by
        # the type of its elements (typek and typev)
        #TODO: review if this can be improved
        typevk, finalvarsk = self._append_values_to_list_after_replace(keyTypesList)
        typevv, finalvarsv = self._append_values_to_list_after_replace(valueTypesList)

        return keyNamesList, valueNamesList, finalvarsk, finalvarsv, typevk, typevv

    def _set_or_tuple(self, type, pk_col, t, t1):
        string_str = ""
        t = t.split(',')
        converted_primary_keys = ", ".join([_conversions.get(w, w) for w in t])
        converted_primary_keys = converted_primary_keys.split(',')
        converted_primary_keys = [w.replace(' ', '') for w in converted_primary_keys]
        aux_list = []  # stores ((var_1, val),(var_2, val),...)
        if len(converted_primary_keys) > 1:
            counter = count(0)
            for type_val in converted_primary_keys:
                if type == "set":
                    aux_list.append((t1 + '_' + str(next(counter)), type_val))
                else:
                    aux_list.append(type_val)
                # string_str = ',{"name": "%s", "type": "%s", "%s": ["%s"]}' % (t1, type, pk_col, '","'.join(aux_list))
                string_str = ',{"name": "%s", "type": "%s", "%s": %s}' % (t1, type, pk_col, aux_list)
        else:
            aux_list.append((t1, converted_primary_keys[0]))
            string_str = ',{"name": "%s", "type": "%s", "%s": %s}' % (t1, type, pk_col, aux_list)
        return string_str

    def _get_dict_str(self, varsk, cleank, typek):
        concatenated_keys = ""
        values = ""
        string_str = ""
        for t, t1, t2 in zip(cleank, varsk, typek):  # first keys
            if t2 == 'set':
                string_str = self._set_or_tuple('set', 'columns', t, t1)
            elif t2 == 'tuple':
                string_str = self._set_or_tuple('tuple', 'columns', t, t1)
            else:
                if t not in _conversions:
                    route = t
                    cname, module = process_path(route)
                    try:
                        mod = __import__(module, globals(), locals(), [cname], 0)
                    except (ImportError, ValueError) as ex:
                        if cname in _conversions:
                            raise Exception("Error parsing the TypeSpec. Maybe you forgot a comma between the columns.")
                        raise ImportError("Can't import class {} from module {}".format(cname, module))
                    string_str = ',("%s", "%s")' % (t1, t)
                else:
                    type = _conversions[t]
                    string_str = ',("%s", "%s")' % (t1, type)
            concatenated_keys = concatenated_keys + string_str
        concatenated_keys = concatenated_keys[1:]
        return concatenated_keys

    def _parse_dict(self, line, this):
        split_line = line.split()
        if len(split_line) == 2:
            pk = split_line[1]
            table = None
        else:
            pk = split_line[2]
            table = split_line[1]
        varsk, varsv, cleank, cleanv, typek, typevv = self._get_str_primary_keys_values(pk)
        pks = self._get_dict_str(varsk, cleank, typek)
        values = self._get_dict_str(varsv, cleanv, typevv)
        if table == None:

            final_dict = '{"primary_keys": [%s], "columns": [%s], "type": "StorageDict"}' % (pks, values)
        else:
            final_dict = '{"%s": {"primary_keys": [%s], "columns": [%s], "type": "StorageDict"}}' % (table, pks, values)
        final_dict = eval(final_dict)
        aux = '{"primary_keys": [%s], "columns": [%s], "type": "StorageDict"}' % (pks, values)
        if table in this:
            this[table].update(eval(aux))
            return this
        return final_dict

    def _parse_set_or_tuple(self, type, line, pk_or_col, this):
        split_line = line.split()
        table = split_line[1]
        line = re.sub('[<>, ]', ' ', split_line[2].replace(str(type), ""))
        primary_keys = line.split()
        converted_primary_keys = ", ".join([_conversions.get(w, w) for w in primary_keys])
        if len(primary_keys) == 1:
            string_str = '{"%s":{"%s": "%s","type": "%s"}}' % (table, pk_or_col, converted_primary_keys, str(type))
            final_string = eval(string_str)
            aux = '{"%s": "%s","type": "%s"}' % (pk_or_col, converted_primary_keys, str(type))
        else:
            string_str = '{"%s":{"%s": "%s","type": "%s"}}' % (table, pk_or_col, converted_primary_keys, str(type))
            final_string = eval(string_str)
            aux = '{"%s": {"%s"},"type": "%s"}' % (pk_or_col, converted_primary_keys, str(type))
        if table in this:
            this[table].update(eval(aux))
            return this
        return final_string

    def _parse_index(self, line, this):
        '''Def: parses index declaration, checking for the introduced vars.
                                Returns: a dict structure with the parsed dict.'''

        if self.type_parser == "TypeSpec":
            table = "indexed_on"
            atributes = line.split(' ', 2)
            atributes = atributes[1].replace(" ", '')
        else:
            table = line.split()[1]
            atributes = line.split(' ', 2)
            atributes = atributes[2].replace(" ", '')

        atributes = atributes.split(',')
        converted_atributes = ", ".join([_conversions.get(w, w) for w in atributes])
        converted_atributes = converted_atributes.split(',')
        converted_atributes = [w.replace(" ", "") for w in converted_atributes]

        if self.type_parser == "TypeSpec":
            this[table] = converted_atributes
        else:
            if table in this:
                this[table].update({'indexed_on': converted_atributes})
            else:
                this[table] = {'indexed_on': converted_atributes}

        return this

    def _parse_file(self, line, new):
        '''Def: Checks if the file declaration is correct.
                Returns: the file declaration with a dict structure'''
        line = line.split(" ")
        output = {}
        table_name = line[1]
        route = line[2]
        cname, module = process_path(route)
        try:
            mod = __import__(module, globals(), locals(), [cname], 0)
        except (ImportError, ValueError) as ex:
            raise ImportError("Can't import class {} from module {}".format(cname, module))
        output["type"] = str(route)
        if table_name in new:
            new[table_name].update(output)
        else:
            new[table_name] = output
        return new

    def _parse_set_tuple_list(self, line, this):
        if line.count('set') > 0:
            return self._parse_set_or_tuple('set', line, 'primary_keys', this)
        elif line.count('tuple') > 0:
            return self._parse_set_or_tuple('tuple', line, 'columns', this)
        elif line.count('list') > 0:
            return self._parse_set_or_tuple('list', line, 'columns', this)

    def _parse_simple(self, line, this):
        split_line = line.split()
        table = split_line[1]
        try:
            type = _conversions[split_line[2]]
        except KeyError as ex:
            raise Exception(f"Type '{split_line[2]}' not identified.")
        simple = '{"%s":{"type":"%s"}}' % (table, type)
        simple = eval(simple)
        if table in this:
            this[table].update(simple)
        return simple

    def _input_type(self, line, this):
        if line.count('<') == 1:  # is tuple, set, list
            aux = self._parse_set_tuple_list(line, this)
        elif line.count('<') == 0 and line.count('Index_on') == 0 and line.count('@stream')==0 and line.count('.') == 0 or (
                line.count('numpy.ndarray') and line.count(' dict') == 0):  # is simple type
            aux = self._parse_simple(line, this)
        elif line.count('Index_on') == 1:
            aux = self._parse_index(line, this)
        elif line.count('.') > 0 and line.count(' dict') == 0:
            aux = self._parse_file(line, this)
        elif line.count('@stream') == 1: #Enable stream
            this['stream']='on'
            aux = this
        else:  # is dict
            aux = self._parse_dict(line, this)
        return aux

    def _remove_spaces_from_line(self, line):
        '''Def: Remove all the spaces of the line splitted from comments
                Returns: same line with no spaces.'''
        line = re.sub(' +', '*', line)
        if line.find('@Index_on') == -1 and line.find('@stream') == -1:
            line = line[line.find(self.type_parser):]

        if line.count('tuple') == 1 and line.count('dict') == 0:
            pos = re.search(r'\b(tuple)\b', line)
            pos = pos.start()
        elif line.count('set') == 1 and line.count('dict') == 0:
            pos = re.search(r'\b(set)\b', line)
            pos = pos.start()
        elif line.count('@stream') == 1:
            pos = line.find('@stream')
            line = line[pos:]
            return line.replace('*', ' ')
        elif line.count('@Index_on') == 1:
            pos = line.find('@Index_on')
            line = line[pos:]
            return line.replace('*', ' ')
        elif line.count('*dict') > 0:
            pos = re.search(r'\b(dict)\b', line)
            pos = pos.start()
        elif line.count('list') > 0:
            pos = re.search(r'\b(list)\b', line)
            pos = pos.start()
        else:
            return line.replace('*', ' ')

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
        self.detect_errors_before(lines, self.type_parser)
        if self.type_parser == "TypeSpec":
            for line in lines.split('\n'):
                this = self._input_type(self._remove_spaces_from_line(line), this)
        if self.type_parser == "ClassField":
            for line in lines.split('\n'):
                this.update(self._input_type(self._remove_spaces_from_line(line), this))
        if self.type_parser == "StreamOnly":
            for line in lines.split('\n'):
                this = self._input_type(self._remove_spaces_from_line(line), this)

        self.detect_errors_after(this, self.type_parser)
        return this

    @staticmethod
    def detect_errors_before(lines, type_parser):
        bad_characters = (";", "&", "(", ")", "[", "]", "=", "?", "¿", "!", "¡")
        # re.escape will escape '|' too, but it shouldn't be escaped, so 'a' is a replacement
        bad_characters = re.escape("a".join(bad_characters)).replace("a", "|")
        bad_found = re.findall(bad_characters, lines)
        if len(bad_found) > 0:
            raise Exception(f"One or more bad character detected: [{', '.join(bad_found)}].")

        if type_parser == "TypeSpec":
            #if len(lines.split("\n")) != 1:
            if len(lines.split("\n")) > 2:
                raise Exception("StorageDicts should only have one TypeSpec line and (optionaly) the Index_on or the stream properties.")
            if lines.count("<") < 2 or lines.count(">") < 2:
                raise Exception("The TypeSpec should have at least two '<' and two '>'. Format: "
                                "@TypeSpec dict<<key:type>, value:type>.")
        elif type_parser == "ClassField":
            for line in lines.split("\n"):
                if ":" in line and "dict" not in line:
                    line_error = line.replace("    ", "")
                    raise Exception(
                        f"The ClassField {line_error} should only have the character ':' if it is in a dict.")

    @staticmethod
    def detect_errors_after(output, type_parser):
        if type_parser == "TypeSpec":
            if "primary_keys" not in output:
                raise Exception("No detected keys. Maybe you forgot to set a primary key or "
                                "there is a missing 'dict' after the TypeSpec.")
            elif "columns" not in output:
                raise Exception("No detected non-key columns.")
        elif type_parser == "ClassField":
            pass

    def __init__(self, type_parser):
        '''Initializes the Parser class with the type_parser that can be @ClassField or @TypeSpec.'''
        self.type_parser = type_parser
