from hecuba.storageobj import StorageObj


class Words(StorageObj):
    '''
    @ClassField words dict<<position:int>,wordinfo:str>
    '''
    pass


class TestStorageObj(StorageObj):
    '''
       @ClassField test dict<<position:int>,text:str>
    '''
    pass