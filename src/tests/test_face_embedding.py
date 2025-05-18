from __future__ import annotations

import unittest
import requests
import numpy as np
from face_recognite.face_embedding import FaceEmbeddingModel
from face_recognite.face_embedding import FaceEmbeddingModelInput, FaceEmbeddingModelOutput
from common.settings import Settings

class TestFaceEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.face_embedding_model = FaceEmbeddingModel(settings=self.settings)

    def process(self, inputs: FaceEmbeddingModelInput) -> FaceEmbeddingModelOutput:
        payload = {
            'image': inputs.image.tolist(),
            'landmarks': inputs.kps.tolist(),
        }
        response = requests.post(str(self.settings.host_embedding), json=payload)
        return FaceEmbeddingModelOutput(embedding=response.json())
    
    def test_embedding(self):
        loaded_embedding = np.load(
            '/home/chien/chien.npy',
        )

        kpoint = np.array([
        [
          348.5597229003906,
          158.63307189941406
        ],
        [
          422.07208251953125,
          162.7647705078125
        ],
        [
          384.935791015625,
          201.4475860595703
        ],
        [
          355.2001647949219,
          240.2054443359375
        ],
        [
          409.2935485839844,
          243.58062744140625
        ]
        ])
        result = self.process(
            inputs=FaceEmbeddingModelInput(
                image=loaded_embedding,
                kps=kpoint,
            ),
        )
        print(result)

if __name__ == '__main__':
    unittest.main()