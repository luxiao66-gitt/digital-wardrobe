import json
from pathlib import Path

from PIL import Image, ImageDraw

from app.services.pipeline.extract_outfit import extract_wardrobe_items


def test_extract_wardrobe_items_writes_clothing_layers_and_json(tmp_path: Path) -> None:
    image_path = tmp_path / "ootd.jpg"
    output_dir = tmp_path / "processed"
    Image.new("RGB", (100, 160), color=(240, 240, 240)).save(image_path)

    def fake_mask_provider(image: Image.Image) -> Image.Image:
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((30, 20, 70, 60), fill=1)
        draw.rectangle((32, 70, 68, 120), fill=2)
        draw.rectangle((10, 125, 90, 150), fill=3)
        return mask

    json_path = extract_wardrobe_items(
        image_path=image_path,
        output_dir=output_dir,
        min_pixels=1,
        mask_provider=fake_mask_provider,
    )

    result = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.exists()
    assert result["extractor"]["type"] == "rembg_cloth_segmentation"
    assert result["extractor"]["saved_item_count"] == 3
    assert [item["category"] for item in result["items"]] == [
        "top",
        "bottom",
        "full_body_outerwear",
    ]
    assert (json_path.parent / "items" / "item_001_top.png").exists()
    assert (json_path.parent / "items" / "item_002_bottom.png").exists()
    assert (json_path.parent / "items" / "item_003_full_body_outerwear.png").exists()


def test_extract_wardrobe_items_skips_empty_layers(tmp_path: Path) -> None:
    image_path = tmp_path / "ootd.jpg"
    output_dir = tmp_path / "processed"
    Image.new("RGB", (100, 160), color=(240, 240, 240)).save(image_path)

    def fake_mask_provider(image: Image.Image) -> Image.Image:
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((30, 20, 70, 60), fill=1)
        return mask

    json_path = extract_wardrobe_items(
        image_path=image_path,
        output_dir=output_dir,
        min_pixels=1,
        mask_provider=fake_mask_provider,
    )

    result = json.loads(json_path.read_text(encoding="utf-8"))

    assert result["extractor"]["saved_item_count"] == 1
    assert result["items"][0]["category"] == "top"
