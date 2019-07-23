from abc import ABC
from typing import List, Tuple, Optional


class TrainableModel(ABC):

    def train(self, data: List[Tuple[str]]):
        raise NotImplementedError

    def get_label(self, input) -> Optional[str]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
