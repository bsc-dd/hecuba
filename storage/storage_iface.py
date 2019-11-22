from abc import ABCMeta, abstractmethod
from uuid import UUID
from storage.cql_iface.tests.mockIStorage import IStorage

class StorageIface(metaclass=ABCMeta):
    @abstractmethod
    def add_data_model(self, definition: dict) -> int:
        """
        Registers a data model describing the data format that will be passed and retrieved from storage.
        :param definition: Describes a data model that will be used to fetch and store data
        :return: data_model_id: Unique identifier to refer to the data model
        """
        pass

    @abstractmethod
    def register_persistent_object(self, datamodel_id: int, pyobject: IStorage) -> UUID:
        """
        Informs the storage that the Hecuba object `pyobject` will be storing and accessing data
        using the data model identified by `datamodel_id`. Returns a unique identifier to identify  the object.
        If the object has been previously registered nothing happens and its ID is returned.
        :param datamodel_id: Identifier of a previously registered data model
        :param pyobject: Hecuba persistent object to register with the persistent storage.
        :return: object_id: UUID to reference and identify the pyobject in the future.

        """
        pass