from __future__ import annotations

import cv2
import numpy as np
import logging
from api.models.face_detector import APIOutput
from fastapi import APIRouter
from fastapi import File
from fastapi import status
from fastapi import UploadFile
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from face_recognite.face_detector import FaceDetectorModel
from face_recognite.face_detector import FaceDetectorModelInput
from common.settings import Settings
from api.helpers.exception_handler import ResponseMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

face_detector = APIRouter(prefix='/v1')

try:
    logger.info('Load mode face detector !!!')
    face_detector_model = FaceDetectorModel(settings=Settings())
except Exception as e:
    logger.error(f'Failed to initialize face embedding model: {e}')
    raise e  # stop and display full error message


@face_detector.post(
    '/detector',
    response_model=APIOutput,
    responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'bboxes': [1, 1, 1, 1],
                            'landmarks': [1, 1, 1, 1, 1],
                        },
                    },
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            'description': 'Bad Request - message is required',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.BAD_REQUEST,
                    },
                },
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            'description': 'Internal Server Error - Error during init conversation',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.INTERNAL_SERVER_ERROR,
                    },
                },
            },
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            'description': 'Unprocessable Entity - Format is not supported',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.UNPROCESSABLE_ENTITY,
                    },
                },
            },
        },
        status.HTTP_404_NOT_FOUND: {
            'description': 'Destination Not Found',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.NOT_FOUND,
                    },
                },
            },
        },
    },
)
async def face_detect(file: UploadFile = File(...)):
    """
    Detects faces in the provided input data.

    Args:
        inputs (FaceDetectorInput): The input data for face detection, which includes image information.
    Returns:
        FaceDetectorOutput: The output data containing detected faces and related details.
    Raises:
        HTTPException: If an error occurs during face detection processing.
    """

    try:
        # Đọc dữ liệu ảnh từ UploadFile
        contents = await file.read()

        # Chuyển dữ liệu ảnh thành mảng numpy
        nparr = np.frombuffer(contents, np.uint8)

        # Giải mã ảnh thành định dạng OpenCV (BGR)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f'Failed to read image data: {e}')
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid image data',
        )

    try:
        # Process image
        response = await face_detector_model.process(
            inputs=FaceDetectorModelInput(
                img=img_array,
            ),
        )
        # handle response
        api_output = APIOutput(
            bboxes=response.bboxes.tolist(),  # type: ignore
            landmarks=response.kpss.tolist(),  # type: ignore
        )
        logger.info('Face detection completed successfully.')
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                'message': ResponseMessage.SUCCESS,
                'info': jsonable_encoder(api_output),
            },
        )
    except Exception as e:
        logger.error(f'Failed to process face detection: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to process face detection',
        )
