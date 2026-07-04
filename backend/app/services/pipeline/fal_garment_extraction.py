from __future__ import annotations

import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DEFAULT_FAL_ENDPOINT = os.getenv(
    "FAL_GARMENT_EXTRACTION_ENDPOINT",
    "fal-ai/fashn/garment-extraction",
)
DEFAULT_IMAGE_ARGUMENT = os.getenv("FAL_GARMENT_IMAGE_ARGUMENT", "image_url")


class FalClientProtocol(Protocol):
    def upload_file(self, path: str) -> str:
        ...

    def subscribe(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


def extract_garments_with_fal(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    endpoint: str = DEFAULT_FAL_ENDPOINT,
    image_argument: str = DEFAULT_IMAGE_ARGUMENT,
    extra_arguments: dict[str, Any] | None = None,
    fal_client_module: FalClientProtocol | None = None,
) -> Path:
    """Call a Fal.ai fashion extraction model and download separated PNG assets.

    The endpoint and image argument name are configurable because Fal model
    schemas vary. For the FASHN garment extraction model, set `endpoint` and
    `image_argument` to the values shown in the Fal playground/API page.
    """
    image_path = Path(image_path).expanduser().resolve()
    output_dir = _resolve_output_dir(output_dir)
    _validate_image_path(image_path)

    fal_client_module = fal_client_module or _load_fal_client()

    photo_id = f"photo_{uuid.uuid4().hex[:12]}"
    run_dir = output_dir / photo_id
    items_dir = run_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    uploaded_url = fal_client_module.upload_file(str(image_path))
    arguments = {
        image_argument: uploaded_url,
        **(extra_arguments or {}),
    }
    result = fal_client_module.subscribe(endpoint, arguments=arguments)
    assets = _extract_image_assets(result)

    items = []
    for index, asset in enumerate(assets, start=1):
        item_id = f"item_{index:03d}"
        category = _normalize_category(asset.get("category") or asset.get("label") or f"garment_{index}")
        asset_url = asset["url"]
        output_path = items_dir / f"{item_id}_{category}.png"
        _download_file(asset_url, output_path)

        items.append(
            {
                "item_id": item_id,
                "category": category,
                "image_path": str(output_path),
                "source_url": asset_url,
                "width": asset.get("width"),
                "height": asset.get("height"),
                "content_type": asset.get("content_type") or _guess_content_type(output_path),
                "tags": {
                    "colors": [],
                    "pattern": None,
                    "style": [],
                    "season": [],
                    "occasion": [],
                },
                "status": "fal_extracted",
                "raw_asset": asset,
            }
        )

    payload = {
        "photo_id": photo_id,
        "source_image": {
            "path": str(image_path),
            "filename": image_path.name,
            "uploaded_url": uploaded_url,
        },
        "extractor": {
            "type": "fal_garment_extraction",
            "endpoint": endpoint,
            "image_argument": image_argument,
            "saved_item_count": len(items),
        },
        "items": items,
        "raw_result": result,
    }

    json_path = run_dir / "result.json"
    _write_json(json_path, payload)
    return json_path


def _load_fal_client() -> FalClientProtocol:
    try:
        import fal_client
    except ImportError as exc:
        raise RuntimeError(
            "fal-client is not installed. Run `pip install -r requirements.txt` "
            "from the backend folder, then try again."
        ) from exc

    if not os.getenv("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set. Set it in your environment before calling Fal.ai.")

    return fal_client


def _extract_image_assets(payload: Any) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    _walk_for_image_assets(payload, assets)
    deduped = []
    seen_urls = set()
    for asset in assets:
        url = asset["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(asset)
    return deduped


def _walk_for_image_assets(value: Any, assets: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        url = value.get("url") or value.get("image_url")
        if isinstance(url, str) and _looks_like_image_url(url):
            asset = dict(value)
            asset["url"] = url
            assets.append(asset)

        for nested in value.values():
            _walk_for_image_assets(nested, assets)
    elif isinstance(value, list):
        for item in value:
            _walk_for_image_assets(item, assets)


def _looks_like_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".png", ".jpg", ".jpeg", ".webp")) or "fal.media" in url


def _download_file(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "digital-wardrobe/0.1"})
    with urlopen(request, timeout=120) as response:
        output_path.write_bytes(response.read())


def _normalize_category(category: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in category)
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "garment"


def _guess_content_type(path: Path) -> str | None:
    return mimetypes.guess_type(path.name)[0]


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
