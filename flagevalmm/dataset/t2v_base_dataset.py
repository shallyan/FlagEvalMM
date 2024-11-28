import json
from typing import Any, Dict
from torch.utils.data import Dataset
from flagevalmm.registry import DATASETS


@DATASETS.register_module()
class Text2VideoBaseDataset(Dataset):
    def __init__(
        self, data_root: str, name: str, debug: bool = False, **kwargs
    ) -> None:
        self.data_root = data_root
        self.name = name
        self.files = json.load(open(data_root))
        self.debug = debug
        if debug:
            self.files = self.files[:32]

    def __len__(self):
        return len(self.files)

    def text_number(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "prompt": self.files[index]["prompt"],
            "id": str(self.files[index]["id"]),
        }

    def get_data(self, index: int) -> Dict[str, Any]:
        assert index < self.text_number()
        return self.__getitem__(index)

    def meta_info(self) -> Dict[str, Any]:
        return {"name": self.name, "length": len(self.files), "type": "t2v"}

    def get_annotation(self) -> Dict[str, Dict[str, Any]]:
        num = self.text_number()
        anno_dict = {}
        for i in range(num):
            data = self.get_data(i)
            anno_dict[data["id"]] = data
        return anno_dict
