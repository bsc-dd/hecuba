import uuid
from . import config, log
from .tools import extract_ks_tab, build_remotely, storage_id_from_name, get_istorage_attrs, generate_token_ring_ranges


class AlreadyPersistentError(RuntimeError):
    pass


class IStorage(object):

    @property
    def storage_id(self):
        return self.__storage_id

    @storage_id.setter
    def storage_id(self, st_id):
        if st_id is not None and not isinstance(st_id, uuid.UUID):
            raise TypeError("Storage ID must be an instance of UUID")
        self.__storage_id = st_id

    def __init__(self, *args, **kwargs):
        super().__init__()
        log.debug('ISTORAGE Init.')
        if not getattr(self, "storage_id", None):
            self.storage_id = kwargs.pop("storage_id", None)
        self._built_remotely = kwargs.pop("built_remotely", False)
        self._tokens = kwargs.pop("tokens", None)
        given_name = kwargs.pop("name", None)
        if given_name:
            try:
                self._ksp, self._table = extract_ks_tab(given_name)
                given_name = self._ksp + '.' + self._table
                # Obtain metas for 'given_name'
                #   metas are the same for all pieces of a splitted object but the storage_id and the tokens.
                #   we get the metas from the entry of the object and for the build_remotely we keep
                #   the storage_id and tokens that we receive in self
                metas = get_istorage_attrs(storage_id_from_name(given_name)) # with split ALL splitted objects have the SAME name, but only 1 has the storage_id corresponding to its name!!
                self._istorage_metas = metas[0] # To pass information retrieved from istorage to subojects to avoid further Cassandra accesses
                given_name   = metas[0].name
                if not self._built_remotely: # When build remotely due to a split, it uses the same table name but with a different storage_id! That is stored later in hecuba istorage.
                    self.storage_id = metas[0].storage_id # Name has priority
                    self._tokens = metas[0].tokens
                if not getattr(self, '_tokens', None): #Tokens are received from kwargs or IStorage otherwise...
                    self._tokens = generate_token_ring_ranges()
            except IndexError:
                pass
        elif self.storage_id:
            try:
                metas = get_istorage_attrs(self.storage_id)
                self._istorage_metas = metas[0] # To pass information retrieved from istorage to subojects to avoid further Cassandra accesses
                given_name   = metas[0].name
                self._tokens = metas[0].tokens
                if not getattr(self, '_tokens', None):
                    self._tokens = generate_token_ring_ranges()
            except IndexError:
                pass
        if given_name :
            # Warning: In order to inherit _tokens and storage_id, they MUST be set before the next call
            IStorage.make_persistent(self, given_name)
        else:
            self._ksp = None
            self._table = None
            self._is_persistent = False


    @classmethod
    def get_by_alias(cls, alias=""):
        return cls(name=alias)

    def __eq__(self, other):
        """
        Method to compare a IStorage object with another one.
        Args:
            other: IStorage to be compared with.
        Returns:
            boolean (true - equals, false - not equals).
        """
        return self.__class__ == other.__class__ and self.storage_id == other.storage_id

    def make_persistent(self, name):
        if getattr(self, '_is_persistent', False):
            raise AlreadyPersistentError("This Object is already persistent [Before:{}.{}][After:{}]".format(
                                         self._ksp, self._table, name))

        self._ksp, self._table = extract_ks_tab(name)
        name = self._ksp + '.' + self._table
        self._set_name(name)

        if not self.storage_id:
            self.storage_id = storage_id_from_name(name)

        # If found data, replace the constructor data
        if not getattr(self,'_tokens', None) :
            metas = get_istorage_attrs(self.storage_id)
            try:
                self._tokens = metas[0].tokens
                self._istorage_metas = metas[0] # To pass information retrieved from istorage to subojects to avoid further Cassandra accesses
            except IndexError:
                self._tokens = generate_token_ring_ranges()

        self._is_persistent = True

    def stop_persistent(self):
        if not self._is_persistent:
            raise RuntimeError("This Object is not persistent")

        self._is_persistent = False

    def delete_persistent(self):
        if not self._is_persistent:
            raise RuntimeError("This Object is not persistent")

        self._is_persistent = False

    def _set_name(self, name):
        if name is not None and not isinstance(name, str):
            raise TypeError("Name -{}-  should be an instance of str".format(str(name)))
        self._name = name

    def _get_name(self):
        try:
            return self._name
        except AttributeError:
            return None

    def sync(self):
        if not self._is_persistent:
            raise RuntimeError("Can't send the data to storage if the object is not persistent")

    def getID(self):
        """
        Method to retrieve the storage id as string. Used by PyCOMPSs solely.
        :return: Storage_id as str
        """
        return str(self.storage_id)

    def split(self, cols=None):
        """
        Method used to divide an object into sub-objects.
        Returns:
            a subobject everytime is called
        """
        from .tools import tokens_partitions
        if cols is not None:
            print("IStorage.split: Ignoring parameter 'cols'. Currently this is only supported for StorageNumpys", flush=True)
        try:
            tokens = self._build_args.tokens
        except AttributeError as ex:
            raise RuntimeError("Object {} does not have tokens".format(self._get_name()))

        self.sync()

        for token_split in tokens_partitions(self._ksp, self._table, tokens):
            storage_id = uuid.uuid4()
            log.debug('assigning to {} num tokens {}'.format(str(storage_id), len(token_split)))
            new_args = self._build_args._replace(tokens=token_split, storage_id=storage_id)
            args_dict = new_args._asdict()
            args_dict["built_remotely"] = True
            yield build_remotely(args_dict)

    def _initialize_stream_capability(self, topic_name=None):
        # Set the stream attributes into 'self' instance

        # Producer and Consumer are lazily created only if 'setitem' or 'poll' are invoked.
        self._stream_producer_enabled = False
        self._stream_consumer_enabled = False

        self._kafka_names = str.join(",", config.kafka_names)
        self._stream_enabled=True
        if topic_name is not None:
            self._topic_name = str(topic_name)
            log.debug("Enabling STREAM %s", self._topic_name)
        else:
            self._topic_name = None

    def _is_stream(self):
        return getattr(self, '_stream_enabled', False)

    def send(self, key=None, value=None):
        raise NotImplementedError("SEND not supported")

    def poll(self):
        raise NotImplementedError("POLL not supported")

