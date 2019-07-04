import uuid
from abc import ABCMeta, abstractmethod
from . import log
from .tools import extract_ks_tab, build_remotely, storage_id_from_name, get_istorage_attrs, generate_token_ring_ranges


class AlreadyPersistentError(RuntimeError):
    pass


class IStorage(metaclass=ABCMeta):

    def get_id(self):
        return self.__storage_id

    def set_id(self, st_id):
        if st_id is not None and not isinstance(st_id, uuid.UUID):
            raise TypeError("Storage ID must be an instance of UUID")
        self.__storage_id = st_id

    storage_id = property(get_id, set_id)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._ksp = ''
        self._table = ''
        if "name" in kwargs:
            self._ksp, self._table = extract_ks_tab(kwargs.pop("name", ''))
            name = self._ksp + '.' + self._table
            self._set_name(name)

        self.storage_id = kwargs.pop("storage_id", None)
        self._built_remotely = kwargs.pop("built_remotely", False)
        self._tokens = kwargs.pop("tokens", [])
        self._is_persistent = False

    def __eq__(self, other):
        """
        Method to compare a IStorage object with another one.
        Args:
            other: IStorage to be compared with.
        Returns:
            boolean (true - equals, false - not equals).
        """
        return self.__class__ == other.__class__ and self.get_id() == other.get_id()


    @abstractmethod
    def make_persistent(self, name):
        self._ksp, self._table = extract_ks_tab(name)
        name = self._ksp + '.' + self._table
        self._set_name(name)

        if not self.storage_id:
            self.storage_id = storage_id_from_name(name)

        # If found data, replace the constructor data
        if not self._tokens:
            metas = get_istorage_attrs(self.storage_id)
            try:
                self._tokens = metas[0].tokens
            except IndexError:
                self._tokens = generate_token_ring_ranges()

        self._is_persistent = True
        self._add_persistent_methods()


    @abstractmethod
    def _stop_persistent(self):
        vars(self)['_stop_persistent'] = vars(self).pop('stop_persistent')
        self.storage_id = None
        self._is_persistent = False


    @abstractmethod
    def _delete_persistent(self):
        vars(self)['_delete_persistent'] = vars(self).pop('delete_persistent')
        self.storage_id = None
        self._is_persistent = False


    def _add_persistent_methods(self):
        vars(self)['delete_persistent'] = self._delete_persistent
        vars(self)['stop_persistent'] = self._stop_persistent


    def _set_name(self, name):
        if not isinstance(name, str):
            raise TypeError("Name -{}-  should be an instance of str".format(str(name)))
        self._name = name


    def _get_name(self):
        try:
            return self._name
        except AttributeError:
            return ''

    def getID(self):
        """
        Method to retrieve the storage id as string. Used by PyCOMPSs solely.
        :return: Storage_id as str
        """
        return str(self.storage_id)


    def split(self):
        """
        Method used to divide an object into sub-objects.
        Returns:
            a subobject everytime is called
        """
        from .tools import tokens_partitions
        try:
            tokens = self._build_args.tokens
            #getattr(self._build_args, 'tokens', None)
        except AttributeError as ex:
            raise RuntimeError("Object {} does not have tokens".format(self._get_name()))

        for token_split in tokens_partitions(self._ksp, self._table, tokens):
            storage_id = uuid.uuid4()
            log.debug('assigning to {} num tokens {}'.format(str(storage_id), len(token_split)))
            new_args = self._build_args._replace(tokens=token_split, storage_id=storage_id)
            args_dict = new_args._asdict()
            args_dict["built_remotely"] = True
            yield build_remotely(args_dict)
