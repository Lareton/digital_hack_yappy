# from qdrant_functions import *
from models import models

from fastapi import FastAPI

app = FastAPI()


@app.post("/check-video-duplicate")
async def check_video(video_link: models.VideoLink):
    return True
