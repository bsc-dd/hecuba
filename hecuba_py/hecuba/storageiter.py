from collections import namedtuple


class NamedIterator:
    # Class that allows to iterate over the keys or the values of a dict
    def __init__(self, hiterator, builder, father):
        self.hiterator = hiterator
        self.builder = builder
        self._storage_father = father

    def __iter__(self):
        return self

    def __next__(self):
        n = self.hiterator.get_next()
        if self.builder is not None:
            if self._storage_father._get_set_types() is not None:
                nkeys = len(n) - len(self._storage_father._get_set_types())
                n = n[:nkeys]
            return self.builder(*n)
        else:
            return n[0]


class NamedItemsIterator:
    # Class that allows to iterate over the keys and the values of a dict
    builder = namedtuple('row', 'key, value')

    def __init__(self, key_builder, column_builder, k_size, hiterator, father):
        self.key_builder = key_builder
        self.k_size = k_size
        self.column_builder = column_builder
        self.hiterator = hiterator
        self._storage_father = father

    def __iter__(self):
        return self

    def __next__(self):
        n = self.hiterator.get_next()
        if self.key_builder is None:
            k = n[0]
        else:
            k = self.key_builder(*n[0:self.k_size])
        if self.column_builder is None:
            v = n[self.k_size]
        else:
            v = self.column_builder(*n[self.k_size:])
        return self.builder(k, v)
