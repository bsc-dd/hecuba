import uuid
from collections import namedtuple


class AlreadyPersistentError(RuntimeError):
    pass


class DataModelNode(object):
    def __init__(self, name=None, class_name=None, args=None):
        self.name = name
        self.class_name = class_name
        self.args = args


def storage_id_from_name(name):
    return uuid.uuid3(uuid.NAMESPACE_DNS, name)


class IStorage(object):
    args_names = ["storage_id"]
    args = namedtuple("IStorage", args_names)
    _build_args = args(storage_id="")

    def getID(self):
        return self.__storage_id

    def setID(self, st_id):
        if st_id is not None and not isinstance(st_id, uuid.UUID):
            raise TypeError("Storage ID must be an instance of UUID")
        self.__storage_id = st_id

    storage_id = property(getID, setID)

    def __new__(cls, name='', *args, **kwargs):
        toret = super(IStorage, cls).__new__(cls)
        toret._ksp = ''
        toret._table = ''
        toret._is_persistent = False
        toret.__storage_id = None
        toret._name = ''
        storage_id = kwargs.get('storage_id', None)

        if storage_id is None and name:
            storage_id = storage_id_from_name(name)

        if name or storage_id:
            toret.setID(storage_id)
            toret.set_name(name)
            toret._is_persistent = True
        return toret

    def __eq__(self, other):
        """
        Method to compare a IStorage object with another one.
        Args:
            other: IStorage to be compared with.
        Returns:
            boolean (true - equals, false - not equals).
        """
        return self.__class__ == other.__class__ and self.getID() == other.getID()

    def is_persistent(self):
        return self.getID() is not None and self.get_name() is not None

    @staticmethod
    def _store_meta(storage_args):
        pass

    def make_persistent(self, name):
        # if not self.storage_id:
        #    self.storage_id = storage_id_from_name(name)
        self._is_persistent = True
        self._name = name

    def stop_persistent(self):
        self.storage_id = None
        self._is_persistent = False

    def delete_persistent(self):
        self.storage_id = None
        self._is_persistent = False

    def split(self):
        """
        Method used to divide an object into sub-objects.
        Returns:
            a subobject everytime is called
        """
        raise NotImplemented("Split not supported yet")

    def set_name(self, name):
        if not isinstance(name, str):
            raise TypeError("Name -{}-  should be an instance of str".format(str(name)))
        self._name = name

    def get_name(self):
        return self._name
