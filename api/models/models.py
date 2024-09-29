from pydantic import BaseModel


class VideoLink(BaseModel):
    link: str


class Response(BaseModel):
    is_duplicate: bool
    duplicate_for: str
