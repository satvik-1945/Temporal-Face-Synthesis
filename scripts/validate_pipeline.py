#!/usr/bin/env python3
"""
Validation script for the face swap pipeline.


For quick testing without real assets, the script can create placeholder images
with solid colors and a message - but face swap will fail (no faces). Use real
photos for full validation.

Usage:
    uv run python scripts/validate_pipeline.py
    # Or with custom paths:
    uv run python scripts/validate_pipeline.py --target path/to/frame.jpg --photo path/to/selfie.jpg
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_placeholder_image(path: Path, label: str, size: tuple = (640, 480)) -> None:
    """Create a simple placeholder image (no face - for testing file I/O only)."""
    import cv2
    import numpy as np

    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = (60, 60, 60)  # Dark gray
    cv2.putText(
        img, label, (50, size[1] // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    cv2.imwrite(str(path), img)
    print(f"Created placeholder: {path} (no face - use real photos for face swap)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Target image (face to replace). If omitted, uses samples/target.jpg",
    )
    parser.add_argument(
        "--photo",
        type=Path,
        default=None,
        help="Source photo (face to use). If omitted, uses samples/source.jpg",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional: validate video pipeline with this video file",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Max frames for video validation (default: 10)",
    )
    parser.add_argument(
        "--model", "-m",
        default="inswapper",
        choices=["inswapper", "ghost", "simswap"],
        help="Face swap model (default: inswapper)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    samples_dir = project_root / "samples"
    samples_dir.mkdir(exist_ok=True)

    target = args.target or samples_dir / "target.jpg"
    photo = args.photo or samples_dir / "source.jpg"

    # Create placeholders if files don't exist
    if not target.exists():
        create_placeholder_image(target, "TARGET (replace with real face image)")
    if not photo.exists():
        create_placeholder_image(photo, "SOURCE (replace with real face photo)")

    print("=" * 60)
    print("Temporal Face Synthesis - Pipeline Validation")
    print("=" * 60)
    print(f"Target: {target}")
    print(f"Source: {photo}")
    print(f"Model: {args.model}")
    print()

    # Step 1: Validate image swap
    print("Step 1: Testing image face swap...")
    try:
        from src.face_swap_pipeline import load_models, swap_face_in_image
        import cv2

        app, swapper, backend = load_models(model_name=args.model)
        print(f"  - Models loaded OK ({args.model})")

        target_img = cv2.imread(str(target))
        source_img = cv2.imread(str(photo))
        if target_img is None or source_img is None:
            raise ValueError("Could not read images")

        result = swap_face_in_image(app, swapper, backend, source_img, target_img)
        output_img = samples_dir / "validation_result.jpg"
        cv2.imwrite(str(output_img), result)
        print(f"  - Face swap OK -> {output_img}")
    except ValueError as e:
        if "No face detected" in str(e):
            print(f"  - No face in images (expected with placeholders). Use real photos!")
        else:
            raise
    except Exception as e:
        print(f"  - FAILED: {e}")
        raise

    # Step 2: Validate video pipeline (if video provided)
    if args.video and args.video.exists():
        print("\nStep 2: Testing video face swap...")
        try:
            from src.face_swap_pipeline import process_video

            output_video = samples_dir / "validation_output.mp4"
            process_video(
                video_path=args.video,
                source_photo_path=photo,
                output_path=output_video,
                max_frames=args.max_frames,
                model_name=args.model,
            )
            print(f"  - Video swap OK -> {output_video}")
        except Exception as e:
            print(f"  - FAILED: {e}")
            raise
    else:
        print("\nStep 2: Skipped (no --video provided)")
        print("  To test video: python scripts/validate_pipeline.py --video path/to/clip.mp4")

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
