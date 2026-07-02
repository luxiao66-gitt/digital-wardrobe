from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ItemTags(BaseModel):
    colors: list[str] = []
    pattern: str | None = None
    style: list[str] = []
    season: list[str] = []
    occasion: list[str] = []


class WardrobeItemResponse(BaseModel):
    item_id: str
    category: str
    bbox: BoundingBox
    image_url: str
    tags: ItemTags

