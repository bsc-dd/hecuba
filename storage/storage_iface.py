from abc import ABCMeta, abstractmethod
from hecuba.IStorage import IStorage
from uuid import UUID
from typing import List, Generator


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
    def delete_persistent_object(self, object_id: UUID) -> bool:
        """
        Deletes the data of the Hecuba object identified by `object_id` from the storage layer.
        :param object_id: Hecuba object identifier
        :return: True on success, False otherwise
        """
        pass

    @abstractmethod
    def add_index(self, datamodel_id: int):
        """

        :param datamodel_id: Identifier of a previously registered data model
        :return: IndexID
        :raises: NotImplemented
        """
        pass

    @abstractmethod
    def get_records(self, object_id: UUID, key_list: List[object]) -> List[object]:
        """
        Returns a list with the records corresponding to the key_list for the Hecuba object referenced by `object_id`.
        :param object_id: Hecuba object identifier
        :param key_list: List with the keys of the records to be retrieved.
        :return: List of the records corresponding to the keys contained in key_list
        """
        pass

    @abstractmethod
    def put_records(self, object_id, key_list: List[object], value_list: List[object]):
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
    def split(self, object_id: UUID) -> Generator[UUID]:
        """
        Partitions the data of the Hecuba object referenced by `object_id` following the same data model.
        Each partition is assigned an UUID.
        :param object_id: Hecuba object identifier
        :return: Yield an `object_id` referencing a subset of the data.
        """
        pass

    @abstractmethod
    def get_data_locality(self, object_id: UUID) -> List[str]:
        """
        Returns a list with the IPs of the nodes holding a copy of the given `object_id`.
        :param object_id: Hecuba object identifier
        :return: List of IPs
        """
        pass
