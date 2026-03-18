# Temporal-Face-Synthesis

AI-powered face swap for video — built for the King movie "feeler" campaign pitch.

Swap faces in videos with a single source photo using InsightFace inswapper_128 (free, open-source).

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

Limit frames for faster testing:

```bash
uv run python main.py swap-video -v template.mp4 -p selfie.jpg -o output.mp4 --max-frames 30
```

## Requirements

- Python 3.12+
- ~2GB disk for models (inswapper_128 + buffalo_l)
- GPU recommended (CUDA) for speed; CPU works

## Project Structure

```
src/
  face_swap_pipeline.py   # Core: load models, swap_face_in_image, process_video
main.py                   # CLI entry point
scripts/
  validate_pipeline.py    # Validation + quick test
samples/                  # Created on first run (placeholder images)
```

## Model

- **inswapper_128.onnx** — Auto-downloaded from Hugging Face to `~/.insightface/models/`
- **buffalo_l** — Face detection/recognition, auto-downloaded by InsightFace
