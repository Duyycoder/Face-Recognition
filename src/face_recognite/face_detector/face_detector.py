from __future__ import annotations

from functools import cached_property

import cv2
import numpy as np
from onnxruntime import InferenceSession  # type: ignore
from common.base import BaseModel
from common.base import BaseService
from common.settings import Settings


class FaceDetectorModelInput(BaseModel):
    img: np.ndarray
    input_size: tuple[int, int] = (640, 640)
    max_num: int = 0
    metric: str = 'default'


class FaceDetectorModelOutput(BaseModel):
    bboxes: np.ndarray
    kpss: np.ndarray | None


class FaceDetectorModel(BaseService):
    settings: Settings

    @cached_property
    def model_loaded(self):
        # Load the face detection model here
        pass

    @cached_property
    def session(self) -> InferenceSession:
        return InferenceSession(self.settings.detector.model_path)

    @cached_property
    def _get_info(self):
        # Get input information
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            input_size = None
        else:
            input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name

        # get output information
        outputs = self.session.get_outputs()
        output_names = []
        for output in outputs:
            output_names.append(output.name)

        center_cache = {}
        num_anchors = self.settings.detector.num_anchors
        use_kps = False
        if len(outputs) == 6:
            fmc = 3
            feat_stride_fpn = [8, 16, 32]
            num_anchors = 2
        elif len(outputs) == 9:
            fmc = 3
            feat_stride_fpn = [8, 16, 32]
            num_anchors = 2
            use_kps = True
        elif len(outputs) == 10:
            fmc = 5
            feat_stride_fpn = [8, 16, 32, 64, 128]
            num_anchors = 1
        elif len(outputs) == 15:
            fmc = 5
            feat_stride_fpn = [8, 16, 32, 64, 128]
            num_anchors = 1
            use_kps = True

        return fmc, feat_stride_fpn, num_anchors, use_kps, input_name, output_names, center_cache, input_size

    async def process(self, inputs: FaceDetectorModelInput) -> FaceDetectorModelOutput:
        """
        Detects faces in an input image using a deep learning model, returning bounding boxes and keypoints.

        Args:
            img (np.ndarray): Input image in BGR format.
            input_size (Tuple[int, int], optional): The target size for the detection model. Defaults to (640, 640).
            max_num (int, optional): The maximum number of faces to return. If 0, all detections are returned.
            metric (str, optional): Sorting metric for selecting faces when max_num > 0. Defaults to 'default'.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - Detected bounding boxes as an array of shape (N, 5) with (x1, y1, x2, y2, score).
                - Keypoints as an array of shape (N, 5, 2) if available, otherwise None.
        """
        # Get model information, including whether keypoints are used
        fmc, feat_stride_fpn, num_anchors, use_kps, input_name, output_names, center_cache, input_size_model = self._get_info

        # Use the model's default input size if none is provided
        assert inputs.input_size is not None or input_size_model is not None
        input_size = input_size_model if inputs.input_size is None else inputs.input_size

        # Compute aspect ratios of the input image and the model's expected input
        im_ratio = float(inputs.img.shape[0]) / inputs.img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]

        # Resize the image while maintaining its aspect ratio
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / inputs.img.shape[0]

        # Resize the input image
        resized_img = cv2.resize(inputs.img, (new_width, new_height))

        # Create a blank image of the model's input size and paste the resized image onto it
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        # Perform face detection using the model
        scores_list, bboxes_list, kpss_list = self.forward(
            det_img, self.settings.detector.conf,
        )

        # Concatenate results from different detection layers
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()

        # Sort detections by confidence score in descending order
        order = scores_ravel.argsort()[::-1]

        # Rescale bounding boxes back to the original image size
        bboxes = np.vstack(bboxes_list) / det_scale

        # Rescale keypoints if they are used
        if use_kps:
            kpss = np.vstack(kpss_list) / det_scale

        # Create a combined array of bounding boxes and confidence scores
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)

        # Sort detections by confidence
        pre_det = pre_det[order, :]

        # Apply Non-Maximum Suppression (NMS) to remove duplicate detections
        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        # Reorder and filter keypoints based on the NMS results
        if use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        # If max_num is specified, keep only the most relevant detections
        if inputs.max_num > 0 and det.shape[0] > inputs.max_num:
            # Compute the area of each bounding box
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])

            # Compute offsets from the image center to prioritize centered faces
            img_center = inputs.img.shape[0] // 2, inputs.img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0],
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            # Determine sorting values based on the chosen metric
            if inputs.metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering

            # Select the top detections based on the computed values
            # some extra weight on the centering
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:inputs.max_num]
            det = det[bindex, :]

            # Filter keypoints accordingly
            if kpss is not None:
                kpss = kpss[bindex, :]

        return FaceDetectorModelOutput(
            bboxes=det,
            kpss=kpss,
        )

    def forward(self, img: np.ndarray, threshold: float) -> tuple:
        """
        Performs a forward pass on the face detection model to extract bounding boxes, keypoints, and confidence scores.

        Args:
            img (np.ndarray): Input image in BGR format as a NumPy array.
            threshold (float): Confidence threshold for filtering detected objects.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                - A list of NumPy arrays containing confidence scores for detected bounding boxes.
                - A list of NumPy arrays containing bounding box coordinates.
                - A list of NumPy arrays containing keypoint coordinates if available.
        """
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(
            img, 1.0/self.settings.detector.input_std, input_size, (
                self.settings.detector.input_mean,
                self.settings.detector.input_mean, self.settings.detector.input_mean,
            ), swapRB=True,
        )

        fmc, feat_stride_fpn, num_anchors, use_kps, input_name, output_names, center_cache, input_size_model = self._get_info
        net_outs = self.session.run(output_names, {input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for idx, stride in enumerate(feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride

            if use_kps:
                kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            _ = height * width
            key = (height, width, stride)

            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                # solution-3:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1,  # type: ignore
                ).astype(np.float32)  # type: ignore

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))

                if num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers]*num_anchors, axis=1,
                    ).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if use_kps:
                kpss = self.distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2+1] + distance[:, i+1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def nms(self, dets):
        thresh = self.settings.detector.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
