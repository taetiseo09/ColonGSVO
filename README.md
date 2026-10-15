# ColonGSVO

__Abstract__: Vision-based odometry systems in colonoscopy aims to reconstruct the colonic lumen and mucosal surface, with the potential for accurate lesion localization. The superior capability in depth and camera pose estimation are required to reconstruct the colon's 3D structures. However, existing methods generally suffer from degraded feature alignment performance across different frames due to reflective surfaces and illumination fluctuations, while also requiring substantial training data rarely available in clinical applications. In this paper, we propose ColonGSVO, a novel self-supervised visual odometry framework to address these issues. Unlike previous approaches, our method leverages 3D Gaussian Splatting to jointly estimate colon depth and colonoscopic camera motion without using ground-truth depth or camera pose during optimization. Specifically, ColonGSVO progressively refines 3D Gaussian representations with our proposed geometric coherence regularization based on optical flow priors. Additionally, to handle illumination variations, we introduce an illumination shift perceiving strategy that achieves robust and finer-grained cross-frame illumination-shift awareness. Experimental results demonstrate that our method achieves competitive depth estimation while improving camera-pose tracking and maintaining high-fidelity 3D reconstruction of the colon.

## Video Demo

[Watch ColonGSVO reconstruction video](assets/colongsvo_sigmoid_t3_a.mp4)

## Requirements

The reference environment uses Ubuntu, Python 3.10, PyTorch 2.0.0, and CUDA 11.7. A CUDA-capable GPU is required for training and evaluation.

```bash
git clone --recursive https://github.com/taetiseo09/ColonGSVO.git
cd ColonGSVO
conda env create -f environment.yml
conda activate colongsvo
```

If the repository is cloned without submodules, initialize them before creating the environment:

```bash
git submodule update --init --recursive
```

The first run downloads DPT and RAFT weights. For offline execution, run once with network access so PyTorch can populate their standard caches, then preserve those caches on the target machine.

## C3VD Data

Each scene should contain an RGB directory and `pose.txt`:

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

## Run the Demo

The following command trains on frames `[0, 300)`, selects the last frame in every eight-frame group, then reports RGB reconstruction quality and rendering FPS.

```bash
./train_c3vd_depth.sh \
  --data-root /path/to/C3VD \
  --scene sigmoid_t3_a \
  --gpu 0 \
  --start 0 \
  --end 300 \
  --split 8
```

Use `./train_c3vd_depth.sh --help` for all options. Two useful modes are:

```bash
# Verify paths and inspect commands without using the GPU.
./train_c3vd_depth.sh --data-root /path/to/C3VD --dry-run

# Evaluate an existing output/colongsvo_demo/... checkpoint.
./train_c3vd_depth.sh --data-root /path/to/C3VD --eval-only
```

The process produces only two report sections:

1. `reconstruction`: PSNR, SSIM, LPIPS, rendered frames, and an MP4 video.
2. `rendering`: synchronized CUDA rendering FPS.

The default output is:

```text
output/colongsvo_demo/<scene>_<start>_<end>/
├── chkpnt/ep00_init.pth
├── run_config.json
├── train/
└── test/
    ├── frames/*.png
    ├── reconstruction.mp4
    ├── report.json
    ├── report.md
    └── run_config.json
```

Use the same `--scene`, `--start`, `--end`, `--split`, and `--expname` values for training and evaluation.

## License

This repository maintains the notices and licenses of previous work. Use and redistribution are limited to non-commercial research or evaluation under [`LICENSE`](LICENSE); Gaussian Splatting components are additionally covered by [`LICENSE_inria.md`](LICENSE_inria.md). Review both files before downloading the repository.
