from __future__ import annotations

import json
from functools import cached_property

import cv2
import numpy as np
from cv2.typing import MatLike
from common.base import BaseModel
from common.settings import Settings
from skimage import transform as trans


class FaceAlignModel(BaseModel):
    settings: Settings

    @cached_property
    def get_map(self) -> tuple[dict, np.ndarray]:
        """
        Load config to get the face alignment mapppin
        """
        with open(self.settings.face_align.file_config_path) as r:
            config = json.load(r)
        src1 = np.array(config['src1'], dtype=np.float32)
        src2 = np.array(config['src2'], dtype=np.float32)
        src3 = np.array(config['src3'], dtype=np.float32)
        src4 = np.array(config['src4'], dtype=np.float32)
        src5 = np.array(config['src5'], dtype=np.float32)

        src = np.array([src1, src2, src3, src4, src5])
        src_map = {112: src, 224: src * 2}
        arcface_src = np.expand_dims(config['arcface_src'], axis=0)
        return src_map, arcface_src

# lmk is prediction; src is template
    def estimate_norm(self, lmk: np.ndarray, image_size: int = 112, mode: str = 'arcface'):
        """
        Estimate the transformation matrix for normalizing the input landmark points to the target landmark points.

        Parameters:
        lmk (np.ndarray): The input landmark points (5x2).
        image_size (int): The target image size (default: 112).
        mode (str): The mode of face alignment (default: 'arcface').

        Returns:
        M (np.ndarray): The transformation matrix.
        pose_index (int): The index of the target landmark points.
        """
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        if mode == 'arcface':
            if image_size == 112:
                src = self.get_map[1]
            else:
                src = float(image_size) / 112 * self.get_map[1]
        else:

            src = self.get_map[0][image_size]
        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]  # type: ignore
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
            #         print(error)
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index

    def norm_crop(self, img: np.ndarray, landmark: np.ndarray, image_size: int = 112, mode: str = 'arcface'):
        """
        Normalize the input landmark points to a square image and crop it to the desired size.

        Parameters:
        img (np.ndarray): The input image.
        landmark (np.ndarray): The input landmark points (5x2).
        image_size (int): The desired image size (default: 112).
        mode (str): The mode of face alignment (default: 'arcface').

        Returns:
        warped (np.ndarray): The cropped and normalized image.
        """
        M, pose_index = self.estimate_norm(landmark, image_size, mode)
        warped = cv2.warpAffine(
            img, M, (image_size, image_size), borderValue=0.0,  # type: ignore
        )
        return warped

    def square_crop(self, im: np.ndarray, S: int):
        """
        Crop the input image to a square image with the specified size.

        Parameters:
        im (np.ndarray): The input image
        S (int): The desired size of the square image

        Returns:
        det_im (np.ndarray): The cropped and resized image
        scale (float): The scale factor used to resize the image
        """
        if im.shape[0] > im.shape[1]:
            height = S
            width = int(float(im.shape[1]) / im.shape[0] * S)
            scale = float(S) / im.shape[0]
        else:
            width = S
            height = int(float(im.shape[0]) / im.shape[1] * S)
            scale = float(S) / im.shape[1]
        resized_im = cv2.resize(im, (width, height))
        det_im = np.zeros((S, S, 3), dtype=np.uint8)
        det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
        return det_im, scale

    def transform(self, data: MatLike, center: list, output_size: int, scale: float, rotation: float):
        """
        Apply 2D transformation to the input image.

        Parameters:
        data (np.ndarray): The input image
        center (list): The center of the transformation (x, y).
        output_size (int): The size of the output image
        scale (float): The scale factor
        rotation (float): The rotation angle in degrees

        Returns:
        cropped (np.ndarray): The transformed image
        M (np.ndarray): The transformation matrix used for the transformation
        """
        scale_ratio = scale
        rot = float(rotation) * np.pi / 180.0
        # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
        t1 = trans.SimilarityTransform(scale=scale_ratio)
        cx = center[0] * scale_ratio
        cy = center[1] * scale_ratio
        t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
        t3 = trans.SimilarityTransform(rotation=rot)
        t4 = trans.SimilarityTransform(
            translation=(output_size / 2, output_size / 2),
        )
        t = t1 + t2 + t3 + t4
        M = t.params[0:2]
        cropped = cv2.warpAffine(
            data,
            M, (output_size, output_size),
            borderValue=0.0,  # type: ignore
        )  # type: ignore
        return cropped, M

    def trans_points2d(self, pts: np.ndarray, M: np.ndarray):
        """
        Transform 2D points using a transformation matrix.

        This function applies a 2D transformation to a set of points using the provided transformation matrix.
        Parameters:
        pts (np.ndarray): An array of 2D points to be transformed.
        Shape should be (n, 2) where n is the number of points.
        M (np.ndarray): A 2x3 transformation matrix.
        Returns:
        np.ndarray: An array of transformed 2D points with the same shape as the input points.
        """
        new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
        for i in range(pts.shape[0]):
            pt = pts[i]
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
            new_pt = np.dot(M, new_pt)
            # print('new_pt', new_pt.shape, new_pt)
            new_pts[i] = new_pt[0:2]

        return new_pts

    def trans_points3d(self, pts: np.ndarray, M: np.ndarray):
        """
        Transform 3D points using a transformation matrix.

        This function applies a 3D transformation to a set of points using the provided transformation matrix.
        Parameters:
        pts (np.ndarray): An array of 3D points to be transformed.
        Shape should be (n, 3) where n is the number of points.
        M (np.ndarray): A 4x4 transformation matrix.
        Returns:
        np.ndarray: An array of transformed 3D points with the same shape as the input points.
        """
        scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
        # print(scale)
        new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
        for i in range(pts.shape[0]):
            pt = pts[i]
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
            new_pt = np.dot(M, new_pt)
            # print('new_pt', new_pt.shape, new_pt)
            new_pts[i][0:2] = new_pt[0:2]
            new_pts[i][2] = pts[i][2] * scale

        return new_pts

    def trans_points(self, pts: np.ndarray, M: np.ndarray):
        """
        Transform 2D or 3D points using a transformation matrix.

        This function applies a 2D or 3D transformation to a set of points using the provided transformation matrix.
        Parameters:
        pts (np.ndarray): An array of points to be transformed.
        Shape should be (n, d) where n is the number of points and d is the dimensionality (2 or 3).
        M (np.ndarray): A transformation matrix.
        Returns:
        np.ndarray: An array of transformed points with the same shape as the input points.
        """
        if pts.shape[1] == 2:
            return self.trans_points2d(pts, M)
        else:
            return self.trans_points3d(pts, M)
