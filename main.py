#!/usr/bin/env python3
"""
CLI for Temporal Face Synthesis - face swap in video.

Usage:
    python main.py swap-video --video path/to/template.mp4 --photo path/to/selfie.jpg --output output.mp4
    python main.py swap-image --target path/to/frame.jpg --photo path/to/selfie.jpg --output result.jpg
"""

import argparse
from pathlib import Path


def cmd_swap_video(args: argparse.Namespace) -> None:
    """Run face swap on video."""
    from src.face_swap_pipeline import process_video

    output = process_video(
        video_path=args.video,
        source_photo_path=args.photo,
        output_path=args.output,
        max_frames=args.max_frames,
        fps_scale=args.fps_scale,
        model_name=args.model,
    )
    print(f"Done! Output saved to: {output}")


def cmd_swap_image(args: argparse.Namespace) -> None:
    """Run face swap on single image (for quick validation)."""
    from src.face_swap_pipeline import load_models, swap_face_in_image
    import cv2

    app, swapper, backend = load_models(model_name=args.model)
    target_img = cv2.imread(str(args.target))
    source_img = cv2.imread(str(args.photo))

    if target_img is None:
        raise FileNotFoundError(f"Could not read: {args.target}")
    if source_img is None:
        raise FileNotFoundError(f"Could not read: {args.photo}")

    result = swap_face_in_image(app, swapper, backend, source_img, target_img)
    cv2.imwrite(str(args.output), result)
    print(f"Done! Output saved to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal Face Synthesis - AI face swap for video"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # swap-video
    p_video = subparsers.add_parser("swap-video", help="Swap face in video")
    p_video.add_argument("--video", "-v", required=True, help="Input video path")
    p_video.add_argument("--photo", "-p", required=True, help="Source face photo")
    p_video.add_argument("--output", "-o", required=True, help="Output video path")
    p_video.add_argument(
        "--model", "-m",
        default="inswapper",
        choices=["inswapper", "ghost", "simswap"],
        help="Face swap model (default: inswapper)",
    )
    p_video.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames for testing (default: all)",
    )
    p_video.add_argument(
        "--fps-scale",
        type=float,
        default=1.0,
        help="Process every Nth frame (1.0=all, 0.5=every other)",
    )
    p_video.set_defaults(func=cmd_swap_video)

    # swap-image
    p_image = subparsers.add_parser("swap-image", help="Swap face in single image")
    p_image.add_argument("--target", "-t", required=True, help="Target image (frame)")
    p_image.add_argument("--photo", "-p", required=True, help="Source face photo")
    p_image.add_argument("--output", "-o", required=True, help="Output image path")
    p_image.add_argument(
        "--model", "-m",
        default="inswapper",
        choices=["inswapper", "ghost", "simswap"],
        help="Face swap model (default: inswapper)",
    )
    p_image.set_defaults(func=cmd_swap_image)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
