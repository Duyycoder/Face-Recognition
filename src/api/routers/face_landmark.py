from __future__ import annotations

import numpy as np
import logging
from api.models.face_landmark import APIInput
from api.models.face_landmark import APIOutput
from fastapi import APIRouter
from fastapi import Body
from fastapi import status
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from face_recognite.face_landmark import FaceLandMark
from face_recognite.face_landmark import FaceLandMarkInput
from api.helpers.exception_handler import ResponseMessage
from common.settings import Settings

# Inits logger and exception handler
# Rename the router to avoid conflict
face_landmark_router = APIRouter(prefix='/v1')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inits face landmark model
try:
    logger.info('Initialize face landmark model')
    face_landmark_service = FaceLandMark(settings=Settings())
except Exception as e:
    logger.error(f"Failed to initialize face landmark service: {e}")
    raise e  # stop and display full error message

# Define API


@face_landmark_router.post(
    '/landmark', response_model=APIOutput, responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'face_angle': [],
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
async def face_landmark(inputs: APIInput = Body(...)):  # Keep the function name as is

    if inputs is None or not inputs.img or not inputs.bbox:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid input data',
        )
    try:
        # Extract embedding
        logger.info('Processing face landmark...')
        landmark = face_landmark_service.process(
            inputs=FaceLandMarkInput(
                img=np.array(inputs.img, dtype=np.uint8),
                bbox=inputs.bbox,
            ),
        )
        result = landmark.pred
        logger.info('Face landmark extracted successfully.')
        return JSONResponse(   
            status_code=status.HTTP_200_OK,
            content={
                'message': ResponseMessage.SUCCESS,
                'info': {
                    'face_angle': jsonable_encoder(result),
                },
            },
        )
    except Exception as e:
        logger.exception(
            f'Exception occurred while processing face landmark: {e}',
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Error processing face landmark',
        )
