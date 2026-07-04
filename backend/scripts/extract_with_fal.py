from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.pipeline.fal_garment_extraction import (
    DEFAULT_FAL_ENDPOINT,
    DEFAULT_IMAGE_ARGUMENT,
    extract_garments_with_fal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract wardrobe items with a Fal.ai fashion model.")
    parser.add_argument("image_path", help="Path to the source OOTD photo.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where downloaded item PNGs and result.json will be written.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_FAL_ENDPOINT,
        help="Fal endpoint, for example the FASHN garment extraction endpoint from Fal's model page.",
    )
    parser.add_argument(
        "--image-argument",
        default=DEFAULT_IMAGE_ARGUMENT,
        help="Input field name the model expects for the uploaded OOTD URL.",
    )
    parser.add_argument(
        "--extra-arguments",
        default=None,
        help="Optional JSON object merged into the Fal model arguments.",
    )
    args = parser.parse_args()

    extra_arguments = json.loads(args.extra_arguments) if args.extra_arguments else None
    json_path = extract_garments_with_fal(
        image_path=Path(args.image_path),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        endpoint=args.endpoint,
        image_argument=args.image_argument,
        extra_arguments=extra_arguments,
    )
    print(json_path)


if __name__ == "__main__":
    main()
