# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Enhanced README** with hero section, performance benchmarks, and real-world use cases
- **Visual demo image** (assets/demo.png) showing RGB input → Raw depth → Refined depth comparison
- **Separate PyPI README** (README_PYPI.md) without HuggingFace YAML frontmatter for clean PyPI display
- **Comprehensive SEO optimization**: 22 PyPI keywords (depth-estimation, robotics, sim2real, realsense, etc.)
- **Enhanced PyPI classifiers**: 13 classifiers including GPU/CUDA, Jupyter, Image Processing topics
- **Demo image generation script** (create_demo_image.py) for marketing assets
- **.gitignore exception** for assets/ directory to track documentation images

### Changed
- **README structure improved**: Added performance table, use cases for robotics/CV/research/production
- **PyPI metadata enhanced**: Better discoverability through robotics, sim2real, and computer vision keywords

### Fixed
- **Ruff F401 linting error**: Removed unused `__version__` import from tests

## [1.0.3] - 2025-01-27

### Added
- **Git LFS support** for HuggingFace Spaces binary files (images)
- **PyPI Trusted Publishing** workflow for automated releases via GitHub tags
- **Example data deployment** to HuggingFace Spaces (color/depth/result images)
- **Automated PyPI publishing** via `.github/workflows/publish-pypi.yml`

### Changed
- **Gradio pinned** to 4.44.1 with pydantic 2.10.6 workaround for HF Spaces compatibility
- **API imports simplified**: `from rgbddepth import RGBDDepth` (updated README and infer.py)
- **xFormers warnings** reduced to debug level (cleaner HF Spaces logs)

### Fixed
- **HuggingFace Spaces deployment**: Git LFS configured for PNG/JPG files
- **PyPI workflow**: Automated publishing on version tags (v*)
- **README documentation**: Corrected Python API import examples
- **Tests**: Removed hardcoded version check to prevent future breakage
- **Import path in infer.py**: Now uses `from rgbddepth import RGBDDepth`

## [1.0.2] - 2025-01-25

### Added
- **xFormers support** for ~8% faster CUDA inference with automatic fallback to SDPA
- **Mixed precision support** (FP16/BF16) via `--precision` flag
- **Device selection** via `--device` flag (auto/cuda/mps/cpu)
- **FlexibleCrossAttention** module for optimized cross-attention with checkpoint compatibility
- **Comprehensive documentation**: README.md with feature comparison, OPTIMIZATION.md guide
- **CI/CD workflows**: Automated tests on push/PR, PyPI publication on release
- **Entry point**: `rgbd-depth` CLI command for easy usage
- **Pre-commit hooks** (black, ruff, isort) for code quality
- **Test suite** with pytest (Python 3.10-3.13 testing)

### Changed
- **Package renamed** from `camera-depth-models` to `rgbd-depth` for PyPI
- **Optimizations auto-enabled** on CUDA by default (use `--no-optimize` to disable)
- **Simplified API**: Removed `OptimizationConfig`, streamlined to `RGBDDepth` with `use_xformers` flag
- **Updated dependencies**: Requires PyTorch 2.0+ for SDPA support
- **Cleaned up codebase**: Removed obsolete scripts, docs, and tests
- **Minimum Python version**: Upgraded to 3.10+

### Fixed
- **Precision preservation**: Pixel-perfect alignment with ByteDance reference (0 pixel diff)
- **Device detection**: Model preprocessing now uses correct device
- **MPS compatibility**: Proper fallback when xFormers/torch.compile not beneficial
- **xFormers cross-attention scaling bug**: Fixed depth prediction accuracy issues
- **Server binding for HF Spaces**: Set `server_name="0.0.0.0"`
- **CUDA depth prediction edge cases**: Added debug logging

### Performance
- **CUDA + xFormers (FP32)**: 0.95s per frame (640×480) — ~8% faster than baseline
- **CUDA + xFormers (FP16)**: 0.52s per frame — ~2× faster than FP32
- **Apple M2 Max (MPS)**: 1.34s per frame — native support, torch.compile disabled
- **CPU (16 cores)**: 13.37s per frame — no GPU required

## [1.0.1] - 2025-01-20

### Added
- **PyPI package publication** as `rgbd-depth`
- **Comprehensive README** with installation instructions
- **Example data** and inference script (`infer.py`)

### Fixed
- **ASCII-only status output** for cross-platform compatibility
- **Ruff linting errors** (F401, E712)
- **Documentation typos**

## [1.0.0] - 2025-01-18

### Added
- **Initial release**: RGBD depth refinement using Vision Transformers
- **Multi-device support**: CUDA, Apple Silicon (MPS), CPU
- **Pre-trained models** for 5 camera types:
  - Intel RealSense D405, D435, L515
  - Stereolabs ZED 2i
  - Microsoft Kinect Azure
- **Reference implementation alignment** with ByteDance Camera Depth Models
- **Modular architecture**: DINOv2 encoder + lightweight DPT decoder
- **Pixel-perfect compatibility**: 0 pixel difference vs ByteDance reference

### Performance
- **Baseline CUDA (FP32)**: ~1.03s per frame (640×480)
- **Baseline MPS**: ~1.34s per frame
- **Baseline CPU**: ~13.37s per frame

---

## Release Notes

### Versioning Strategy
- **Major (X.0.0)**: Breaking API changes, architectural overhauls
- **Minor (1.X.0)**: New features, camera support, model checkpoints
- **Patch (1.0.X)**: Bug fixes, dependency updates, deployment improvements

### Upgrade Guide

#### From 1.0.2 to 1.0.3
No breaking changes. Simply upgrade:
```bash
pip install --upgrade rgbd-depth
```

#### From 1.0.1 to 1.0.2
- **Requires Python 3.10+** (dropped 3.8, 3.9 support)
- No API changes

### Known Issues

#### HuggingFace Spaces
- **xFormers warnings**: xFormers is not available on HF Spaces (CPU-only environment). This is expected and does not affect functionality.
- **Gradio version**: Must use exactly `gradio==4.44.1` due to JSON schema bug in 4.44.0.

#### PyPI Trusted Publishing
- **Manual configuration required** at https://pypi.org/manage/account/publishing/
- See `.github/workflows/publish-pypi.yml` for workflow setup

### Links
- **GitHub Repository**: https://github.com/Aedelon/camera-depth-models
- **PyPI Package**: https://pypi.org/project/rgbd-depth/
- **HuggingFace Spaces**: https://huggingface.co/spaces/Aedelon/rgbd-depth
- **Original Research**: https://manipulation-as-in-simulation.github.io/
- **ByteDance Reference**: https://github.com/bytedance/camera-depth-models

---

**Note:** This package maintains 100% compatibility with original ByteDance checkpoints.
All weights are interchangeable between this optimized version and the reference implementation.
