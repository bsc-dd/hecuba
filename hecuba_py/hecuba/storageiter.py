from .IStorage import IStorage
import storage


class ImplStorageIter:
    def __new__(cls, *args, **kwargs):
        toret = super().__new__(cls)
        storage_id = kwargs.get('storage_id', None)
        # toret.data = storage.StorageAPI.get_records(storage_id,[])
        return toret

    def __next__(self):
        if self.data is None:
            raise StopIteration()

        for elem in self.data:
            yield elem
        self.data = []
        raise StopIteration()


class StorageIter(IStorage):
    _persistent_props = None
    _persistent_attrs = None

    def __new__(cls, *args, **kwargs):
        toret = super().__new__(cls, *args, **kwargs)
        storage_id = kwargs.get('storage_id', None)
        persistent_props = kwargs.pop('data_model', None)
        name = kwargs.pop('name', '')
        toret.set_name(name)
        DataModelId = storage.StorageAPI.add_data_model(persistent_props)

        if storage_id:
            toret.storage_id = storage_id
            toret._is_persistent = True
            toret.myiter = ImplStorageIter(storage_id=storage_id)
            storage.StorageAPI.register_persistent_object(DataModelId, toret)
        else:
            toret.myiter = iter([])

        return toret

    def __iter__(self):
        return self

    def __next__(self):
        # and transform to namedtuple
        return next(self.myiter)
