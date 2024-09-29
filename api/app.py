# from qdrant_functions import *
from models import models

from fastapi import FastAPI


def test(link: str):
    return True, "something"


app = FastAPI()


@app.post("/check-video-duplicate")
async def check_video(video_link: models.VideoLink) -> models.Response:
    is_duplicate, duplicate_for = test(video_link.link)
    return models.Response(is_duplicate=is_duplicate, duplicate_for=duplicate_for)
