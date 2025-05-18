from __future__ import annotations

import numpy as np
import logging
from api.models.face_embedding import APIInput
from api.models.face_embedding import APIOutput
from fastapi import APIRouter
from fastapi import Body
from fastapi import status
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from face_recognite.face_embedding import FaceEmbeddingModel
from face_recognite.face_embedding import FaceEmbeddingModelInput
from common.settings import Settings
from api.helpers.exception_handler import ResponseMessage

face_embedding = APIRouter(prefix='/v1')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the face embedding model
try:
    logger.info('Initializing face embedding model !!')
    face_embedding_service = FaceEmbeddingModel(settings=Settings())
except Exception as e:
    logger.error(f'Failed to initialize face embedding model: {e}')
    raise e  # stop and display full error message

# Define API input

@face_embedding.post(
    '/embedding',
    response_model=APIOutput,
    responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'face_embedding': [],
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
async def face_embed(inputs: APIInput = Body(...)):

    if inputs is None or not inputs.image or not inputs.landmarks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid input data',
        )
    try:
        # Extract embedding
        logger.info('Processing face embedding...')
        logger.info(f'image shape: {np.array(inputs.image).shape}')

        embedding = face_embedding_service.process(
            inputs=FaceEmbeddingModelInput(
                image=np.array(inputs.image, dtype=np.uint8),
                kps=np.array(inputs.landmarks, dtype=np.float64),
            ),
        )
        logger.info('Face embedding extracted successfully.')
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                'message': ResponseMessage.SUCCESS,
                'info': {
                    'face_embedding': jsonable_encoder(str(embedding.embedding)),
                },
            },
        )
    except Exception as e:
        logger.exception(
            f'Exception occurred while processing face embedding: {e}',
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Error processing face embedding',
        )
