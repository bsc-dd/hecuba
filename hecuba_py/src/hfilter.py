from collections import Iterable

import regex
import inspect
from hecuba import config
# from hecuba.qbeast import QbeastIterator, QbeastMeta

from IStorage import IStorage
from hecuba.hdict import NamedItemsIterator

magical_regex = regex.compile(r'(?:\d+(?:\.\d+)?|\w|"\w+")+|[^\s\w\_]')


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
        elif (elem != 'in' and not isinstance(elem, int) and not regex.match(r'[^\s\w]', elem)) and not elem.isdigit():
            i = elem.find('.')
            if i > 0:
                elem_var = elem[:i]
                if elem_var not in func_vars:
                    elemm = elem[i:]
                    get_ele = dictv.get(str(elemm))
                    if get_ele is None:
                        list_with_values.append(elem)
                    else:
                        list_with_values.append(dictv.get(str(elem)))
                else:
                    list_with_values.append(elem[i+1:])
            else:
                get_elem = dictv.get(str(elem), elem)
                list_with_values.append(get_elem)
        else:
            list_with_values.append(elem)

    return list_with_values


def is_float(var):
    try:
        float(var)
        return True
    except ValueError:
        return False


def transform_to_correct_type(final_list, dictv):
    final = []
    reverse_comparison = {">=": "<=", "<=": ">=", ">": "<", "<": ">"}
    for elem in final_list:
        aux = []
        for i, value in enumerate(elem):
            if isinstance(value, (int, float, Iterable)) and not isinstance(value, str):
                aux.append(value)
            elif not value.find('"') == -1:
                aux.append(value.replace('"', ''))
            elif value.isdigit() and value not in dictv.values():
                aux.append(int(value))
            elif is_float(value) and value not in dictv.values():
                aux.append(float(value))
            elif value == "True":
                aux.append(True)
            elif value == "False":
                aux.append(False)
            else:
                aux.append(value)

        if (isinstance(aux[0], str) and aux[0].isdigit()) or isinstance(aux[0], int):
            aux.reverse()
            aux[1] = reverse_comparison[aux[1]]

        final.append(aux)

    return final


def parse_lambda(func):
    func_vars, clean_string = func_to_str(func)
    parsed_string = magical_regex.findall(clean_string)
    simplified_filter = []

    for i, elem in enumerate(parsed_string):
        if i > 0:
            if elem == '=' and simplified_filter[-1] == "=":
                pass
            elif elem == '=' and (simplified_filter[-1] == "<" or simplified_filter[-1] == ">"):
                simplified_filter[-1] = simplified_filter[-1] + "="
            elif simplified_filter[-1][-1] == ".":
                simplified_filter[-1] += elem
            elif elem == ".":
                simplified_filter[-1] = simplified_filter[-1] + elem
            else:
                simplified_filter.append(elem)
        else:
            simplified_filter.append(elem)

    # Getting variables
    dictv = {}
    for i, elem in enumerate(func.__code__.co_freevars):
        dictv[str(elem)] = func.__closure__[i].cell_contents

    # Combine set or tuple
    for i, elem in enumerate(simplified_filter):
        if elem is "[":
            index = simplified_filter[i:].index(']')
            c = ''.join(simplified_filter[i:index + i + 1])
            simplified_filter[i:index + i + 1] = [eval(c)]
        elif elem is '(':
            index = simplified_filter[i:].index(')')
            c = ''.join(simplified_filter[i:index + i + 1])
            simplified_filter[i:index + i + 1] = [eval(c)]

    # Creating sublists
    lastpos = 0
    newpos = 0
    final_list = []
    if len(simplified_filter) > 3:
        while newpos < len(simplified_filter):
            if 'and' in simplified_filter[lastpos:]:
                newpos = simplified_filter[lastpos:].index('and')
                newpos = newpos + lastpos
            else:
                newpos = len(simplified_filter)
            sublist = simplified_filter[lastpos:newpos]
            if () in sublist:
                sublist.remove(())
            lastpos = newpos + 1
            sublist = substit_var(sublist, func_vars, dictv)

            final_list.append(sublist)
    else:

        sublist = substit_var(simplified_filter, func_vars, dictv)
        final_list.append(sublist)
    # Replace types for correct ones

    final_list = transform_to_correct_type(final_list, dictv)
    # print(final_list)
    return final_list


def hfilter(lambda_filter, iterable):
    if not isinstance(iterable, IStorage):
        try:
            iterable = iterable._storage_father
        except AttributeError:
            return python_filter(lambda_filter, iterable)

    parsed_lambda = parse_lambda(lambda_filter)

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
