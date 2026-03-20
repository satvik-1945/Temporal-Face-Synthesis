# Temporal-Face-Synthesis

AI-powered face swap for video — built for the King movie "feeler" campaign pitch.

Swap faces in videos with a single source photo. Supports multiple models:
- **inswapper** (default) — Fast, 128px, InsightFace
- **ghost** — Higher quality, 256px (ai-forever)
- **simswap** — Higher quality, 256px (neuralchen)

## Quick Start

### 1. Install

```bash
uv sync
# or: pip install -e .
```

### 2. Validate Pipeline

First run validates model download (~600MB) and loads FaceAnalysis + inswapper:

```bash
uv run python scripts/validate_pipeline.py
```

With real photos for full face swap test:

```bash
uv run python scripts/validate_pipeline.py --target path/to/frame.jpg --photo path/to/selfie.jpg
```

With video (processes first 10 frames by default):

```bash
uv run python scripts/validate_pipeline.py --video path/to/clip.mp4 --photo path/to/selfie.jpg
```

### 3. CLI Usage

**Image face swap** (quick test):

```bash
uv run python main.py swap-image --target frame.jpg --photo selfie.jpg --output result.jpg
```

**Video face swap**:

```bash
uv run python main.py swap-video --video template.mp4 --photo selfie.jpg --output output.mp4
```

**Choose model** (ghost/simswap for better quality):

```bash
uv run python main.py swap-image -t frame.jpg -p selfie.jpg -o result.jpg --model ghost
uv run python main.py swap-video -v template.mp4 -p selfie.jpg -o output.mp4 --model simswap
```

Limit frames for faster testing:

```bash
uv run python main.py swap-video -v template.mp4 -p selfie.jpg -o output.mp4 --max-frames 30
```

## Requirements

- Python 3.12+
- ~2.5GB disk for all models (inswapper + ghost + simswap + buffalo_l)
- GPU recommended (CUDA) for speed; CPU works

## Project Structure

```
src/
  face_swap_pipeline.py   # Core: load models, swap_face_in_image, process_video
  swappers/               # Face swap backends
    inswapper.py          # InsightFace 128px (fast)
    ghost.py              # Ghost 256px (higher quality)
    simswap.py            # SimSwap 256px (higher quality)
main.py                   # CLI entry point
scripts/
  validate_pipeline.py    # Validation + quick test
samples/                  # Created on first run (placeholder images)
```

## Models

Models are auto-downloaded to `~/.insightface/models/` on first use:

- **inswapper_128.onnx** — Default, fast, 128px
- **ghost_1_256.onnx** + **crossface_ghost.onnx** — Ghost 256px (FaceFusion)
- **simswap_256.onnx** + **crossface_simswap.onnx** — SimSwap 256px (FaceFusion)
- **buffalo_l** — Face detection/recognition (InsightFace)
