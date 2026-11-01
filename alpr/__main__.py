"""Command line entry point: python -m alpr IMAGE [--output DIR]."""

from __future__ import annotations

import argparse
import json

import cv2

from .pipeline import recognize_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Find and read an Indian vehicle registration plate.")
    parser.add_argument("image", help="Path to a vehicle photograph")
    parser.add_argument("--output", default="alpr_output", help="Directory for intermediate images and results.json")
    parser.add_argument("--max-candidates", type=int, default=15, help="Number of plate proposals to inspect")
    parser.add_argument("--show", action="store_true", help="Display the annotated detection in an OpenCV window")
    args = parser.parse_args()

    result = recognize_image(args.image, args.output, args.max_candidates)
    print(json.dumps({
        "plate_number": result.plate_number,
        "valid_indian_format": result.valid_indian_format,
        "confidence": result.confidence,
        "ocr_method": result.ocr_method,
        "plate_box_xywh": result.plate_box,
        "output_dir": args.output,
    }, indent=2))
    if args.show:
        cv2.imshow("License plate recognition", result.annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
