from __future__ import annotations

import json
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
CLOTHING_LAYERS = {
    1: "top",
    2: "bottom",
    3: "full_body_outerwear",
}
MaskProvider = Callable[[Image.Image], Image.Image]


def extract_wardrobe_items(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    min_pixels: int = 500,
    mask_provider: MaskProvider | None = None,
) -> Path:
    """Extract individual clothing-layer PNGs from an OOTD photo.

    The rembg `u2net_cloth_seg` model returns a clothing parse mask where:
    0 = background, 1 = top, 2 = bottom, 3 = full-body / outerwear.

    Each non-empty clothing layer is isolated into a transparent PNG and
    auto-cropped to remove empty padding. A `result.json` file is written next
    to the item PNGs and its path is returned.
    """
    image_path = Path(image_path).expanduser().resolve()
    output_dir = _resolve_output_dir(output_dir)
    _validate_image_path(image_path)

    photo_id = f"photo_{uuid.uuid4().hex[:12]}"
    run_dir = output_dir / photo_id
    items_dir = run_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGBA")
        width, height = image.size
        raw_mask = _get_cloth_segmentation_mask(image, mask_provider=mask_provider)
        mask = _normalize_mask_map(raw_mask, size=image.size)

        items = []
        for layer_value, category in CLOTHING_LAYERS.items():
            alpha = _build_layer_alpha(mask, layer_value=layer_value)
            bbox = alpha.getbbox()
            if bbox is None:
                continue

            pixel_count = _count_mask_pixels(alpha)
            if pixel_count < min_pixels:
                continue

            isolated = Image.new("RGBA", image.size, (0, 0, 0, 0))
            isolated.paste(image, (0, 0), alpha)
            cropped = isolated.crop(bbox)

            item_id = f"item_{len(items) + 1:03d}"
            crop_path = items_dir / f"{item_id}_{category}.png"
            cropped.save(crop_path)

            items.append(
                {
                    "item_id": item_id,
                    "category": category,
                    "mask_value": layer_value,
                    "bbox": {
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                    },
                    "pixel_count": pixel_count,
                    "image_path": str(crop_path),
                    "tags": {
                        "colors": [],
                        "pattern": None,
                        "style": [],
                        "season": [],
                        "occasion": [],
                    },
                    "status": "cloth_segmentation_extracted",
                    "notes": "Extracted from rembg u2net_cloth_seg clothing mask.",
                }
            )

        result = {
            "photo_id": photo_id,
            "source_image": {
                "path": str(image_path),
                "filename": image_path.name,
                "format": original.format,
                "width": width,
                "height": height,
            },
            "extractor": {
                "type": "rembg_cloth_segmentation",
                "model": "u2net_cloth_seg",
                "mask_values": {
                    "0": "background",
                    "1": "top",
                    "2": "bottom",
                    "3": "full_body_outerwear",
                },
                "min_pixels": min_pixels,
                "saved_item_count": len(items),
            },
            "items": items,
        }

    json_path = run_dir / "result.json"
    _write_json(json_path, result)
    return json_path


def extract_outfit_items_to_json(
    image_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Backward-compatible wrapper for the current wardrobe extractor."""
    return extract_wardrobe_items(image_path=image_path, output_dir=output_dir)


def _get_cloth_segmentation_mask(
    image: Image.Image,
    mask_provider: MaskProvider | None = None,
) -> Image.Image:
    if mask_provider is not None:
        return mask_provider(image)

    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise RuntimeError(
            "rembg is not installed. Run `pip install -r requirements.txt` "
            "from the backend folder, then try extraction again."
        ) from exc

    session = new_session("u2net_cloth_seg", providers=["CPUExecutionProvider"])
    mask = remove(image, session=session, only_mask=True)
    if isinstance(mask, Image.Image):
        return mask
    return Image.open(BytesIO(mask))


def _normalize_mask_map(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    mask = ImageOps.exif_transpose(mask)
    if mask.size != size:
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)

    if mask.mode not in {"L", "P"}:
        mask = mask.convert("L")
    elif mask.mode == "P":
        mask = mask.convert("L")

    extrema = mask.getextrema()
    if extrema[1] <= 3:
        return mask

    # Some mask outputs are encoded as 0/85/170/255 rather than 0/1/2/3.
    return mask.point(lambda value: min(3, int(round(value / 85))))


def _build_layer_alpha(mask: Image.Image, layer_value: int) -> Image.Image:
    return mask.point(lambda value: 255 if value == layer_value else 0).convert("L")


def _count_mask_pixels(alpha: Image.Image) -> int:
    histogram = alpha.histogram()
    return sum(histogram[1:])


def _validate_image_path(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image type '{image_path.suffix}'. Use one of: {supported}")


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[4]
    return project_root / "data" / "processed"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
