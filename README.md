# CF-3DGS for C3VD

This repository extends [NVlabs/CF-3DGS](https://github.com/NVlabs/CF-3DGS) with a reproducible C3VD pipeline for COLMAP-free endoscopic reconstruction. The C3VD demo adds:

- C3VD image/pose loading and configurable train/test splits;
- DPT monocular-depth initialization with optional inverse depth;
- PTLFlow optical-flow supervision and intensity masking;
- pose, novel-view-synthesis, and runtime evaluation;
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

The following command trains on frames `[0, 300)`, holds out the last frame in every four-frame group, then evaluates pose and novel-view synthesis:

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

The default output is:

```text
output/c3vd_inverse_depth_demo/<scene>_<start>_<end>/
├── chkpnt/ep00_init.pth
├── pose/
├── eval/
├── train/
└── timing_*.json
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

## Reproducibility notes

- The C3VD demo uses `DPT_Hybrid` through `torch.hub` and UniMatch through PTLFlow (`mix` checkpoint).
- Generated weights, visualizations, data, logs, and caches are ignored by Git.
- `DepthAnything/` and `RAFT/` are not vendored. The released C3VD demo does not depend on those local checkouts.
- The code records initialization, tracking, mapping, per-frame, and aggregate timing statistics in the experiment directory.

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
