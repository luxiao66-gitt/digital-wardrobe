from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ItemTags(BaseModel):
    colors: list[str] = Field(default_factory=list)
    pattern: str | None = None
    style: list[str] = Field(default_factory=list)
    season: list[str] = Field(default_factory=list)
    occasion: list[str] = Field(default_factory=list)


class WardrobeItemResponse(BaseModel):
    item_id: str
    category: str
    bbox: BoundingBox
    image_url: str
    tags: ItemTags
