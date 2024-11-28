import json
from typing import Dict
from torch.utils.data import Dataset
from flagevalmm.registry import DATASETS


@DATASETS.register_module()
class Text2ImageBaseDataset(Dataset):
    def __init__(
        self, data_root: str, name: str, debug: bool = False, **kwargs
    ) -> None:
        self.data_root = data_root
        self.name = name
        self.data = json.load(open(data_root))
        self.debug = debug
        if debug:
            self.data = self.data[:32]

    def __len__(self) -> int:
        return len(self.data)

    def text_number(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return {"prompt": self.data[index]["prompt"], "id": str(self.data[index]["id"])}

    def get_data(self, index: int):
        assert index < self.text_number()
        return self.__getitem__(index)

    def meta_info(self):
        return {"name": self.name, "length": len(self.data), "type": "t2i"}

    def get_annotation(self):
        num = self.text_number()
        anno_dict = {}
        for i in range(num):
            data = self.get_data(i)
            anno_dict[data["id"]] = data
        return anno_dict
