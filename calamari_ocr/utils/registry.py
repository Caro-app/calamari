class DataRegistry(object):

    def __init__(self) -> None:
        self._obj_map = {}

    def register(self, index: int, name: str, size: int):
        self._obj_map[index] = {'name': name, 'size': size}

    def get_name(self, index: int) -> str:
        ret_name = self._obj_map.get(index)
        if ret_name is None:
            raise KeyError(f"No object indexed {index}"
                           "found in data registry!")
        return ret_name.get('name')

    def get_size(self, index: int) -> str:
        ret_name = self._obj_map.get(index)
        if ret_name is None:
            raise KeyError(f"No object indexed {index}"
                           "found in data registry!")
        return ret_name.get('size')

    def __contains__(self, index: int) -> bool:
        return index in self._obj_map


dataregistry = DataRegistry()
