# CF-3DGS for C3VD

This repository extends [NVlabs/CF-3DGS](https://github.com/NVlabs/CF-3DGS) with a reproducible C3VD pipeline for COLMAP-free endoscopic reconstruction. The C3VD demo adds:

- C3VD image/pose loading and configurable train/test splits;
- DPT monocular-depth initialization with optional inverse depth;
- PTLFlow optical-flow supervision and intensity masking;
- RGB reconstruction reporting, video export, and rendering-FPS evaluation;
- a parameterized train/evaluate script with no machine-specific paths.

The repository intentionally excludes datasets, checkpoints, generated results, model caches, and local third-party checkouts.

## Requirements

The reference environment uses Ubuntu, Python 3.10, PyTorch 2.0.0, and CUDA 11.7. A CUDA-capable GPU is required for training and evaluation.

```bash
git clone --recursive https://github.com/wuyou012/CF-3DGS-C3VD.git
cd CF-3DGS-C3VD
conda env create -f environment.yml
conda activate cf3dgs-c3vd
```

If the repository was cloned without submodules, initialize them before creating the environment:

```bash
git submodule update --init --recursive
```

The first run downloads MiDaS/DPT and PTLFlow weights. For offline execution, run once with network access so PyTorch and PTLFlow can populate their standard caches, then preserve those caches on the target machine.

## C3VD data layout

Each scene must contain an RGB directory and `pose.txt`:

```text
/path/to/C3VD/
└── sigmoid_t3_a/
    ├── color/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    ├── pose.txt
    └── camera.json          # optional
```

`pose.txt` must have one line per image and 16 comma-separated values per line. Images are naturally sorted, and the number of images must match the number of poses.

The loader uses the reference C3VD intrinsics by default. To override them for a converted or resized sequence, add `camera.json` to the scene directory:

```json
{
  "image_height": 540,
  "image_width": 675,
  "fx": 401.1595,
  "fy": 400.9425,
  "cx": 334.143,
  "cy": 273.8665,
  "png_depth_scale": 2.55,
  "crop_edge": 0
}
```

## Run the demo

The following command trains on frames `[0, 300)`, holds out the last frame in every four-frame group, then reports RGB reconstruction quality and rendering FPS. Depth and pose tests are not part of the evaluation pipeline.

```bash
./train_c3vd_depth.sh \
  --data-root /path/to/C3VD \
  --scene sigmoid_t3_a \
  --gpu 0 \
  --start 0 \
  --end 300 \
  --split 4
```

Use `./train_c3vd_depth.sh --help` for all options. Two useful modes are:

```bash
# Verify paths and inspect commands without using the GPU.
./train_c3vd_depth.sh --data-root /path/to/C3VD --dry-run

# Evaluate an existing output/c3vd_inverse_depth_demo/... checkpoint.
./train_c3vd_depth.sh --data-root /path/to/C3VD --eval-only
```

The evaluation produces only two report sections:

1. `reconstruction`: PSNR, SSIM, LPIPS, rendered frames, and an MP4 video.
2. `rendering`: synchronized CUDA rendering FPS.

The default output is:

```text
output/c3vd_inverse_depth_demo/<scene>_<start>_<end>/
├── chkpnt/ep00_init.pth
├── train/
└── test/
    ├── frames/*.png
    ├── reconstruction.mp4
    ├── report.json
    └── report.md
```

Use the same `--scene`, `--start`, `--end`, `--split`, and `--expname` values for training and evaluation. The demo script enforces this automatically.

## Direct Python entry point

For custom experiments, the equivalent training command is:

```bash
python run_cf3dgs.py \
  --source /path/to/C3VD/sigmoid_t3_a \
  --mode train \
  --data_type tanks \
  --images color \
  --depth_model_type dpt \
  --use_flow_loss \
  --expname c3vd_inverse_depth_demo \
  --start_cam 0 \
  --end_cam 300 \
  --train_test_split 4 \
  --inverse_depth \
  --intensity_threshold 0.5
```

`--data_type tanks` is retained for compatibility with the original trainer. Dataset dispatch detects C3VD from `pose.txt`.

To evaluate an existing checkpoint with the dedicated two-section evaluator:

```bash
./evaluate_c3vd.sh \
  --source /path/to/C3VD/sigmoid_t3_a \
  --model-path output/c3vd_inverse_depth_demo/sigmoid_t3_a_0_300/chkpnt/ep00_init.pth \
  --start 0 \
  --end 300 \
  --split 4 \
  --video-fps 10
```

## Example reconstruction report

The included `sigmoid_t3_a`, frames `[0, 300)`, split-8 result contains 37 held-out views:

- Reconstruction: **PSNR 31.681, SSIM 0.853, LPIPS 0.297**.
- Rendering FPS: **299.21**.
- [Download the reconstruction video](assets/c3vd_sigmoid_t3_a_reconstruction.mp4).
- [View the machine-readable report](assets/c3vd_sigmoid_t3_a_report.json).

The reported FPS measures only synchronized Gaussian rendering. Metric computation, PNG writing, and MP4 encoding are excluded.

## Reproducibility notes

- The C3VD demo uses `DPT_Hybrid` through `torch.hub` and UniMatch through PTLFlow (`mix` checkpoint).
- Generated weights, visualizations, data, logs, and caches are ignored by Git.
- `DepthAnything/` and `RAFT/` are not vendored. The released C3VD demo does not depend on those local checkouts.
- Monocular depth remains part of training initialization, but no depth metric or depth image is produced by evaluation.
- Pose estimates remain an internal reconstruction input, but no pose metric or pose report is produced.

## Upstream project and citation

This work is derived from COLMAP-Free 3D Gaussian Splatting. Please cite the upstream paper when using the code:

```bibtex
@InProceedings{Fu_2024_CVPR,
  author    = {Fu, Yang and Liu, Sifei and Kulkarni, Amey and Kautz, Jan and Efros, Alexei A. and Wang, Xiaolong},
  title     = {COLMAP-Free 3D Gaussian Splatting},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {20796--20805}
}
```

## License

This derivative repository retains the upstream notices and licenses. Use and redistribution are limited to non-commercial research or evaluation under [`LICENSE`](LICENSE); Gaussian Splatting components are additionally covered by [`LICENSE_inria.md`](LICENSE_inria.md). Review both files before sharing the repository with collaborators or making it public.
