export type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type ItemTags = {
  colors: string[];
  pattern?: string;
  style: string[];
  season: string[];
  occasion: string[];
};

export type WardrobeItem = {
  itemId: string;
  category: string;
  bbox: BoundingBox;
  imageUrl: string;
  tags: ItemTags;
};

