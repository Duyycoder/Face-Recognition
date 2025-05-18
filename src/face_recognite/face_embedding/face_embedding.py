from __future__ import annotations

from functools import cached_property
from typing import Any

import cv2
import numpy as np
import onnx
from onnx import GraphProto
from onnx import ModelProto
from onnxruntime import InferenceSession  # type: ignore
from common.base import BaseModel
from common.base import BaseService
from common.settings import Settings

from ..face_align import FaceAlignModel



class FaceEmbeddingModelInput(BaseModel):
    image: np.ndarray
    kps: np.ndarray


class FaceEmbeddingModelOutput(BaseModel):
    embedding: Any


class FaceEmbeddingModel(BaseService):
    settings: Settings

    @cached_property
    def face_align(self) -> FaceAlignModel:
        return FaceAlignModel(settings=self.settings)

    @cached_property
    def model_loaded(self) -> ModelProto:
        return onnx.load(self.settings.embedding.model_path)

    @cached_property
    def model_graph(self) -> GraphProto:
        return self.model_loaded.graph

    @cached_property
    def session(self) -> InferenceSession:
        return InferenceSession(self.settings.embedding.model_path)

    @cached_property
    def _get_info(self) -> dict:
        find_sub, find_mul = False, False
        for _, node in enumerate(self.model_graph.node[:8]):
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            input_mean = self.settings.embedding.input_mean_available
            input_std = self.settings.embedding.input_std_available
        else:
            input_mean = self.settings.embedding.input_mean_unavailable
            input_std = self.settings.embedding.input_std_unavailable

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name

        input_size = tuple(input_shape[2:4][::-1])
        outputs = self.session.get_outputs()

        output_names = []
        for out in outputs:
            output_names.append(out.name)
        output_shape = outputs[0].shape

        model_info: dict = {
            'input_size': input_size,
            'input_shape': input_shape,
            'input_name': input_name,
            'output_names': output_names,
            'output_shape': output_shape,
            'input_mean': input_mean,
            'input_std': input_std,
        }
        return model_info

    def process(self, inputs: FaceEmbeddingModelInput) -> FaceEmbeddingModelOutput:
        aimg = self.face_align.norm_crop(inputs.image, landmark=inputs.kps)
        embedding = self._get_feat(aimg).flatten()
        return FaceEmbeddingModelOutput(embedding=embedding)

    def _get_feat(self, imgs):
        """
        Process images and return face embeddings using the loaded ONNX model.

        Parameters:
        imgs (Union[np.ndarray, List[np.ndarray]]): The input images (single image or list of images).

        Returns:
        np.ndarray: The face embeddings for the input images.
        """
        if not isinstance(imgs, list):
            imgs = [imgs]

        model_info = self._get_info

        blob = cv2.dnn.blobFromImages(
            imgs, 1.0 / model_info['input_std'], model_info['input_size'], (
                model_info['input_mean'], model_info['input_mean'], model_info['input_mean'],
            ), swapRB=True,
        )

        net_output = self.session.run(
            model_info['output_names'],
            {model_info['input_name']: blob},
        )[0]
        return net_output
