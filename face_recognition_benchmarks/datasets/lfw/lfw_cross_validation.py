from itertools import chain
from typing import List, Tuple, Any, Iterator

from face_recognition_benchmarks.base import TrainableModel
from face_recognition_benchmarks.datasets.lfw.lfw_parser import classified_data
from face_recognition_benchmarks.datasets.lfw.paths import base

arranged_dataset = classified_data()


def load_folds() -> List[List[Tuple[str, str]]]:
    folds_spec = base / 'people.txt'

    with folds_spec.open() as f:
        folds = [[] for _ in range(int(f.readline()))]

        for fold in folds:
            num_samples = int(f.readline())

            for _ in range(num_samples):
                label, index = f.readline().split()
                fold.append((arranged_dataset[label][int(index) - 1], label))

        return folds


def cross_validate(model: TrainableModel) -> float:

    def skip_nth(target: List[List[Any]], n: int) -> Iterator[Any]:
        return chain(*(target[:n] + target[n+1:]))

    folds = load_folds()
    accuracy = []

    for skip in range(len(folds)):
        model.train(list(skip_nth(folds, skip)))

        recognized = 0

        for image, label in folds[skip]:
            if model.get_label(image) == label:
                recognized += 1

        accuracy.append(recognized / len(folds[skip]))

    return sum(accuracy) / len(accuracy)
