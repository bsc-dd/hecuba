from collections import Iterable

import regex
import inspect
import re
from hecuba import config
from hecuba.qbeast import QbeastIterator, QbeastMeta

from IStorage import IStorage
from hecuba.tools import NamedItemsIterator

magical_regex = regex.compile('(?:\d+(?:\.\d+)?|\w|"\w+")+|[^\s\w\_]')


def func_to_str(func):
    func_string = inspect.getsourcelines(func)[0][0]
    start, end = func_string.find("lambda"), func_string.rfind(",")
    func_string = func_string[start:end]
    func_vars = func_string[7:func_string.find(':')].replace(" ", "").split(',')
    clean_string = func_string[func_string.find(':') + 1:].replace("\\n", '').replace("'", '')
    return func_vars, clean_string


def substit_var(final_list, func_vars, dictv):
    list_with_values = []
    for elem in final_list:
        if not isinstance(elem, str) and isinstance(elem, Iterable):
            list_with_values.append(elem)
        elif (elem != 'in' and not isinstance(elem, int) and not regex.match('[^\s\w]', elem)) and (not elem.isdigit()):
            if elem.find('.') > 0:
                elem_var = elem[:elem.find('.')]
                if elem_var not in func_vars:
                    elemm = elem[elem.find('.'):]
                    get_ele = dictv.get(str(elemm))
                    if get_ele is None:
                        list_with_values.append(elem)
                    else:
                        list_with_values.append(dictv.get(str(elem)))
                else:
                    list_with_values.append(elem[elem.find('.') + 1:])
            else:
                get_elem = dictv.get(str(elem), elem)
                list_with_values.append(get_elem)
        else:
            list_with_values.append(elem)

    return list_with_values


def istype(var):
    try:
        int(var)
        return 'int'
    except ValueError:
        try:
            float(var)
            return 'float'
        except ValueError:
            return 'str'


def transform_to_correct_type(final_list, dictv):
    final = []
    for elem in final_list:
        aux = []
        for i, value in enumerate(elem):
            if isinstance(value, int) or isinstance(value, list) or isinstance(value, set) \
                    or isinstance(value, tuple) or isinstance(value, float):
                aux.append(value)
            elif not value.find('"') == -1:
                aux.append(value.replace('"', ''))
            elif value.isdigit() and value not in dictv.values():
                aux.append(int(value))
            elif istype(value) is 'float' and value not in dictv.values():
                aux.append(float(value))
            elif re.match('True', value) is not None:
                aux.append(True)
            elif re.match('False', value) is not None:
                aux.append(False)
            else:
                aux.append(value)

        if (isinstance(aux[0], str) and aux[0].isdigit()) or isinstance(aux[0], int):
            aux.reverse()
            if aux[1] == '>=':
                aux[1] = '<='
            elif aux[1] == '<=':
                aux[1] = '>='
            elif aux[1] == '>':
                aux[1] = '<'
            elif aux[1] == '<':
                aux[1] = '>'
        final.append(aux)

    return final


def parse_lambda(func):
    func_vars, clean_string = func_to_str(func)
    parsed_string = magical_regex.findall(clean_string)
    for i, elem in enumerate(parsed_string):
        try:
            float(elem)
        except ValueError:
            try:
                if elem.index('.') > -1:
                    parsed_string[i - 1:i + 2] = [''.join(parsed_string[i - 1:i + 2])]
            except:
                if (elem is '=') and re.match('(>|<)', parsed_string[i - 1]) is not None:
                    parsed_string[i - 1:i + 1] = [''.join(parsed_string[i - 1:i + 1])]
                elif (elem is '=') and (parsed_string[i - 1] is '='):
                    parsed_string[i - 1:i + 1] = '='

    # Getting variables
    dictv = {}
    for i, elem in enumerate(func.__code__.co_freevars):
        dictv[str(elem)] = func.__closure__[i].cell_contents

    # Combine set or tuple
    for i, elem in enumerate(parsed_string):
        if elem is "[":
            index = parsed_string[i:].index(']')
            c = ''.join(parsed_string[i:index + i + 1])
            parsed_string[i:index + i + 1] = [eval(c)]
        elif elem is '(':
            index = parsed_string[i:].index(')')
            c = ''.join(parsed_string[i:index + i + 1])
            parsed_string[i:index + i + 1] = [eval(c)]

    # Creating sublists
    lastpos = 0
    newpos = 0
    final_list = []
    if len(parsed_string) > 3:
        while newpos < len(parsed_string):
            if 'and' in parsed_string[lastpos:]:
                newpos = parsed_string[lastpos:].index('and')
                newpos = newpos + lastpos
            else:
                newpos = len(parsed_string)
            sublist = parsed_string[lastpos:newpos]
            if () in sublist:
                sublist.remove(())
            lastpos = newpos + 1
            sublist = substit_var(sublist, func_vars, dictv)

            final_list.append(sublist)
    else:

        sublist = substit_var(parsed_string, func_vars, dictv)
        final_list.append(sublist)
    # Replace types for correct ones

    final_list = transform_to_correct_type(final_list, dictv)
    # print(final_list)
    return final_list


def hfilter(lambda_filter, iterable):
    if not isinstance(iterable, IStorage):
        return python_filter(lambda_filter, iterable)

    parsed_lambda = parse_lambda(lambda_filter)

    if isinstance(iterable, QbeastIterator):
        non_index_arguments = ""
        # initialize lists of the same size as indexed_args
        from_p = [None] * len(iterable._indexed_args)
        to_p = [None] * len(iterable._indexed_args)
        precision = None

        for expression in parsed_lambda:
            if expression[0] in iterable._indexed_args:
                index = iterable._indexed_args.index(expression[0])
                if expression[1] == ">":
                    from_p[index] = expression[2]
                elif expression[1] == "<":
                    to_p[index] = expression[2]
                elif expression[1] == "in":
                    raise Exception("Cannot use <in> on a QbeastIterator")
                else:
                    non_index_arguments += "%s %s %s AND " % (expression[0], expression[1], expression[2])
            elif expression[0].find("random") > -1:
                precision = expression[2]
            else:
                non_index_arguments += "%s %s %s AND " % (expression[0], expression[1], expression[2])

        if precision is None:
            precision = 1.0
        name = "%s.%s" % (iterable._ksp, iterable._table)
        qbeast_meta = QbeastMeta(non_index_arguments[:-5], from_p, to_p, precision)
        new_iterable = QbeastIterator(primary_keys=iterable._primary_keys, columns=iterable._columns,
                                      indexed_args=iterable._indexed_args, name=name, qbeast_meta=qbeast_meta,
                                      tokens=iterable._tokens)
        return new_iterable

    if not iterable._is_persistent:
        raise Exception("The StorageDict needs to be persistent.")

    predicate = Predicate(iterable)
    for expression in parsed_lambda:
        if expression[1] in (">", "<", "=", ">=", "<="):
            predicate = predicate.comp(col=expression[0], comp=expression[1], value=expression[2])
        elif expression[1] == "in":
            predicate = predicate.inside(col=expression[0], values=expression[2])
        else:
            raise Exception("Bad expression.")

    return predicate.execute()


class Predicate:
    def __init__(self, father):
        self.father = father
        self.primary_keys = [name for (name, _) in self.father._primary_keys]
        self.columns = [name for (name, _) in self.father._columns]
        self.predicate = None

    def comp(self, col, value, comp):
        '''
        Select all rows where col (==, >=, <=, >, <) value
        '''
        if col not in self.columns + self.primary_keys:
            raise Exception("Wrong column.")

        if self.predicate is not None:
            self.predicate += " AND "
        else:
            self.predicate = ""

        if isinstance(value, str):
            value = "'{}'".format(value)

        self.predicate += " {} {} {}".format(col, comp, value)
        return self

    def inside(self, col, values):
        '''
        Select all rows where col in values
        '''
        if col not in self.primary_keys:
            raise Exception("Column not in primary key.")

        if self.predicate is not None:
            self.predicate += " AND "
        else:
            self.predicate = ""

        self.predicate += " {} IN (".format(col)
        for value in values:
            if isinstance(value, str):
                value = "'{}'".format(value)
            self.predicate += "{}, ".format(value)
        self.predicate = self.predicate[:-2] + ")"
        return self

    def execute(self):
        '''
        Execute the CQL query
        Returns an iterator over the rows
        '''
        conditions = self.predicate + " ALLOW FILTERING"

        hiter = self.father._hcache.iteritems({'custom_select': conditions, 'prefetch_size': config.prefetch_size})
        iterator = NamedItemsIterator(self.father._key_builder,
                                      self.father._column_builder,
                                      self.father._k_size,
                                      hiter,
                                      self.father)

        return iterator
