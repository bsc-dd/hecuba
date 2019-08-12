from abc import ABCMeta, abstractmethod


class StorageIface(metaclass=ABCMeta):
    @abstractmethod
    def add_data_model(self, definition):
        # datamodel_id
        pass

    @abstractmethod
    def register_persistent_object(self, datamodel_id, pyobject):
        # object_id
        pass

    @abstractmethod
    def delete_persistent_object(self, object_id):
        # bool
        pass

    @abstractmethod
    def add_index(self, datamodel_id):
        # IndexID
        pass

    @abstractmethod
    def get_records(self, object_id, key_list):
        # List < Value >
        pass

    @abstractmethod
    def put_records(self, object_id, key_list, value_list):
        # uint64[] ?
        pass

    @abstractmethod
    def split(self, object_id):
        # List < object_id >
        pass

    @abstractmethod
    def get_data_locality(self, object_id):
        # List < Node >
        pass
