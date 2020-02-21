from collections import namedtuple


class NamedIterator:
    def __init__(self, hiterator, builder):
        self.hiterator = hiterator
        self.builder = builder

    def __iter__(self):
        return self

    def __next__(self):
        n = self.hiterator.get_next()[0]
        return n


class NamedItemsIterator:
    builder = namedtuple('row', 'key, value')

    def __init__(self, key_builder, column_builder, k_size, hiterator):
        self.key_builder = key_builder
        self.k_size = k_size
        self.column_builder = column_builder
        self.hiterator = hiterator

    def __iter__(self):
        return self

    def __next__(self):
        n = self.hiterator.get_next()
        k = self.key_builder(*n[0:self.k_size])
        v = self.column_builder(*n[self.k_size:])
        return self.builder(k, v)
