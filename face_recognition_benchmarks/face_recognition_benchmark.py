from typing import Optional, List, Tuple

import face_recognition

from face_recognition_benchmarks.base import TrainableModel
from face_recognition_benchmarks.datasets.lfw import lfw_cross_validation


# TODO: Tuple[str] -> Own type
class FaceRecognitionTrainable(TrainableModel):
    def __init__(self):
        self.encodings = []

    def train(self, data: List[Tuple[str]]):
        for file, label in data:
            image = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                self.encodings.append((label, encodings[0]))

    def get_label(self, input) -> Optional[str]:
        results = face_recognition.compare_faces([encoding for _, encoding in self.encodings])

        for index, result in enumerate(results):
            if result:
                return self.encodings[index][0]

        return None

    def reset(self) -> None:
        self.encodings = []


if __name__ == '__main__':
    result = lfw_cross_validation.cross_validate(FaceRecognitionTrainable())
    print(result)