from __future__ import annotations

from typing import List

import cv2
import numpy as np
import onnx
from face_recognite.face_align import FaceAlignModel
from onnx import GraphProto
from onnx import ModelProto
from onnxruntime import InferenceSession  # type: ignore
from common.base import BaseModel
from common.base import BaseService
from common.settings import Settings

from .utils import convert_106p_to_86p
from .utils import get_angle
from .utils import get_cameraMatrix
from .utils import get_line
from .utils import ref3DModel


class FaceLandMarkInput(BaseModel):
    img: np.ndarray
    bbox: list


class FaceLandMarkOutput(BaseModel):
    pred: List[float]


class ModelInfo(BaseModel):
    input_size: tuple
    input_shape: tuple
    input_name: str
    output_names: list
    output_shape: tuple
    input_mean: float
    input_std: float
    lmk_dim: int
    lmk_num: int
    taskname: str


class FaceLandMark(BaseService):
    settings: Settings

    @property
    def face_align(self) -> FaceAlignModel:
        return FaceAlignModel(settings=self.settings)

    @property
    def face3Dmodel(self):
        return ref3DModel()

    @property
    def session(self) -> InferenceSession:
        return InferenceSession(self.settings.landmark.model_path)

    @property
    def model_loaded(self) -> ModelProto:
        return onnx.load(self.settings.landmark.model_path)

    @property
    def model_graph(self) -> GraphProto:
        return self.model_loaded.graph

    @property
    def _get_info(self) -> ModelInfo:
        """
        This function retrieves and processes information about the ONNX model used for face landmark detection.
        It determines the input size, input shape, input name, output names, output shape, input mean, input standard deviation,
        landmark dimension, landmark number, and task name based on the ONNX model properties
        .
        Returns:
        ModelInfo: An object containing the retrieved and processed information about the ONNX model.
        """
        find_sub = False
        find_mul = False
        for nid, node in enumerate(self.model_graph.node[:8]):
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
            if nid < 3 and node.name == 'bn_data':
                find_sub = True
                find_mul = True
        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = self.settings.landmark.input_mean_available
            input_std = self.settings.landmark.input_std_available
        else:
            input_mean = self.settings.landmark.input_mean_unavailable
            input_std = self.settings.landmark.input_std_unavailable

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name

        input_size = tuple(input_shape[2:4][::-1])
        outputs = self.session.get_outputs()

        output_names = []
        for out in outputs:
            output_names.append(out.name)
        output_shape = outputs[0].shape

        if output_shape[1] == 3309:
            lmk_dim = 3
            lmk_num = 68
        else:
            lmk_dim = 2
            lmk_num = output_shape[1]//lmk_dim
        taskname = 'landmark_%dd_%d' % (lmk_dim, lmk_num)

        model_info = ModelInfo(
            input_size=input_size,
            input_shape=input_shape,
            input_name=input_name,
            output_names=output_names,
            output_shape=output_shape,
            input_mean=input_mean,
            input_std=input_std,
            lmk_dim=lmk_dim,
            lmk_num=lmk_num,
            taskname=taskname,
        )
        return model_info

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get_face_angle(self, image, landmark, draw=True):
        """
        Calculate the face angle using 3D face model and landmark points.

        This function estimates the face orientation in 3D space using a set of 2D facial landmarks
        and a predefined 3D face model. It uses the solvePnP algorithm to compute the rotation
        and translation vectors, which are then decomposed to obtain the face angles.

        Parameters:
        image (numpy.ndarray): The input image containing the face.
        landmark (numpy.ndarray): An array of facial landmark points.
        draw (bool, optional): If True, draws the landmark points on the image. Defaults to True.

        Returns:
        numpy.ndarray: An array of three angles (pitch, yaw, roll) representing the face orientation in 3D space.

        Note:
        The function uses specific landmark points (indices 86, 0, 35, 93, 52, 61) for angle calculation.
        The returned angles are in degrees.
        """
        if draw:
            for la in landmark:
                cv2.circle(image, la.astype(int), 1, (155, 155, 155), 1)
        refImgPts = np.array(
            [
                landmark[86], landmark[0], landmark[35],
                landmark[93], landmark[52], landmark[61],
            ], dtype=np.float64,
        )
        height, width, channel = image.shape
        focalLength = width
        cameraMatrix = get_cameraMatrix(focalLength, (height / 2, width / 2))
        mdists = np.zeros((4, 1), dtype=np.float64)
        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(
            self.face3Dmodel, refImgPts, cameraMatrix, mdists,
        )
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(
            noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists,
        )
        # p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        # p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))

        return angles  # , p1,p2

    def get_face_angle2(self, image, landmark):
        """
        Calculate the face angle using facial landmarks.

        This function estimates the face angle by computing the angle between
        a perpendicular line and the nose midline using facial landmarks.

        Parameters:
        image (numpy.ndarray): The input image containing the face.
        landmark (numpy.ndarray): An array of 106 facial landmark points.

        Returns:
        float: The calculated face angle in degrees.
        """
        dlib_face_landmark = convert_106p_to_86p(landmark)
        perp_line, _, _, _, _ = get_line(
            dlib_face_landmark, image, type='perp_line',
        )
        nose_mid_line, _, _, _, _ = get_line(
            dlib_face_landmark, image, type='nose_long',
        )

        angle = get_angle(perp_line, nose_mid_line)

        return angle

    def process(self, inputs: FaceLandMarkInput) -> FaceLandMarkOutput:
        """
        Processes a facial image to extract landmark points.

        This function takes an input image containing a face and its bounding box (bbox),
        aligns the face, normalizes the image, and runs a landmark detection model
        to extract facial landmark coordinates.

        Args:
            inputs (FaceLandMarkInput): An object containing the input image and
                                        bounding box information of the face.

        Returns:
            FaceLandMarkOutput: An array containing the coordinates of facial landmarks
                                after alignment and processing.
        """
        w, h = (
            inputs.bbox[2] - inputs.bbox[0]
        ), (inputs.bbox[3] - inputs.bbox[1])
        center = (inputs.bbox[2] + inputs.bbox[0]) / \
            2, (inputs.bbox[3] + inputs.bbox[1]) / 2
        rotate = 0

        model_info = self._get_info

        _scale = model_info.input_size[0] / (max(w, h)*1.5)
        aimg, M = self.face_align.transform(
            inputs.img, list(
                center,
            ), model_info.input_size[0], _scale, rotate,
        )
        input_size = tuple(aimg.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            aimg, 1.0/model_info.input_std, input_size,
            (
                model_info.input_mean, model_info.input_mean,
                model_info.input_mean,
            ), swapRB=True,
        )
        pred = self.session.run(
            model_info.output_names, {
                model_info.input_name: blob,
            },
        )[0][0]
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if model_info.lmk_num < pred.shape[0]:
            pred = pred[model_info.lmk_num*-1:, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (model_info.input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (model_info.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        align = self.face_align.trans_points(pred, IM)
        pred = self.get_face_angle(image=inputs.img, landmark=align)
        return FaceLandMarkOutput(pred=pred)
