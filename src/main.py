from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.face_detector import face_detector
from api.routers.face_embedding import face_embedding
from api.routers.face_landmark import face_landmark_router

app = FastAPI(title="Face checkin", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(face_detector)
app.include_router(face_embedding)
app.include_router(face_landmark_router)