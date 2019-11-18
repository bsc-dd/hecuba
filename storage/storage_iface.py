from abc import ABCMeta, abstractmethod

class StorageIface(metaclass=ABCMeta):
    @abstractmethod
    def add_data_model(self, definition: dict) -> int:
        """
        Registers a data model describing the data format that will be passed and retrieved from storage.
        :param definition: Describes a data model that will be used to fetch and store data
        :return: data_model_id: Unique identifier to refer to the data model
        """
        pass
