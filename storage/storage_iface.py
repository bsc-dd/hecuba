from abc import ABCMeta, abstractmethod
from typing import Generator
from uuid import UUID
from collections import OrderedDict
from typing import List
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

    @abstractmethod
    def put_record(self, object_id: UUID, key_list: OrderedDict, value_list: OrderedDict) -> None:
        """
        Stores the records contained in value_list, which correspond to the keys in key_list
        for the Hecuba object referenced by `object_id`.
        :param object_id: Hecuba object identifier
        :param key_list: List with the keys of the records to be stored.
        :param value_list: List with the records to be stored.
        :return: -
        """
        pass

    @abstractmethod
    def get_record(self, object_id: UUID, key_list: OrderedDict) -> List[object]:
        """
        Returns a list with the records corresponding to the key_list for the Hecuba object referenced by `object_id`.
        :param object_id: Hecuba object identifier
        :param key_list: List with the keys of the records to be retrieved.
        :return: List of the records corresponding to the keys contained in key_list
        """
        pass

    @abstractmethod
    def split(self, object_id: UUID, subsets: int) -> Generator[UUID, UUID, None]:
        """
        Partitions the data of the Hecuba object referenced by `object_id` following the same data model.
        Each partition is assigned an UUID.
        :param object_id: Hecuba object identifier
        :return: Yield an `object_id` referencing a subset of the data.
        """
        pass

