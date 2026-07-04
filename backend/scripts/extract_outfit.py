from __future__ import annotations

import argparse
from pathlib import Path

from app.services.pipeline.extract_outfit import extract_wardrobe_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract wardrobe items from an outfit photo.")
    parser.add_argument("image_path", help="Path to the source outfit photo.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where cropped items and result.json will be written.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=500,
        help="Ignore clothing mask layers smaller than this many pixels.",
    )
    args = parser.parse_args()

    json_path = extract_wardrobe_items(
        image_path=Path(args.image_path),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        min_pixels=args.min_pixels,
    )
    print(json_path)


if __name__ == "__main__":
    main()
