<h1 align="center">GlassFormer</h1>
<p align="center">
  <b>Learning Real-time Glass Segmentation using Radar-Depth Fusion</b><br>
  Suhani Grover, Astik Srivastava, Viswas Dinesh, Avinash Sharma, Madhava Krishna
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/paper-IROS%202026-blue"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://suhani92.github.io/GlassFormer/"><img src="https://img.shields.io/badge/project-page-orange"></a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="90%" alt="GlassFormer teaser">
</p>

> Transparent surfaces are a persistent failure case for robotic perception: RGB
> cameras see the background behind glass, and depth sensors return invalid or
> background measurements at transparent interfaces. **GlassFormer** fuses
> 60.5 GHz millimeter-wave radar with RGB-D sensing. Radar reflects strongly off
> glass precisely where vision and depth fail, giving a geometric cue that is
> insensitive to lighting. We turn the radar–depth inconsistency into a coarse
> spatial prior and inject it into a SegFormer-B2 backbone via cross-modal
> **RadarAttention**, achieving **0.88 mIoU** on a mixed-condition split and
> **0.59 mIoU** on a dedicated low-light split, in real time on
> resource-constrained platforms.

---

## Highlights

- **Radar-guided mask generation** — fuses mmWave range peaks with RGB-D depth
  inconsistencies to localize transparent-surface candidates in the image plane.
- **GlassFormer** — a SegFormer-B2 network with lightweight `RadarAttention`
  modules at the two deepest encoder stages; robust under low-light and glare.
- **Synchronized RGB-D-radar dataset** — 1800 annotated frames spanning glass
  doors, windows, walls, mirrors, and tinted glass across daylight-to-near-dark.
- **Open-source ROS2 driver** for the Acconeer XM125 pulsed coherent radar, plus
  a real-time inference node.
- Runs at **70.6 fps** on an RTX-4060 laptop GPU and **13.2 fps** CPU-only.

## Results

**Well-lit (mixed) split**

| Method | mIoU ↑ | MAE ↓ | F-measure ↑ | BER ↓ |
|---|---|---|---|---|
| GDNet | 0.5485 | 0.2894 | 0.6951 | 0.2618 |
| GlassSemNet | 0.6336 | 0.1945 | 0.7769 | – |
| SegFormer | 0.8799 | 0.0938 | 0.9361 | 0.0517 |
| Radar Overlay | 0.4687 | 0.3474 | 0.6382 | 0.3348 |
| **GlassFormer (ours)** | **0.8818** | **0.0835** | **0.9372** | **0.0517** |

**Low-light split**

| Method | mIoU ↑ | MAE ↓ | F-measure ↑ | BER ↓ |
|---|---|---|---|---|
| GDNet | 0.4824 | 0.4031 | 0.682 | 0.3981 |
| GlassSemNet | 0.4623 | 0.3620 | 0.7010 | – |
| SegFormer | 0.4912 | 0.3588 | 0.6588 | 0.2819 |
| Radar Overlay | 0.5407 | 0.3148 | 0.7019 | 0.3097 |
| **GlassFormer (ours)** | **0.5904** | **0.2895** | **0.7424** | **0.2406** |

## Repository structure

```
GlassFormer/
├── glassformer/                    # ML package (model, data, train, eval)
│   ├── models/glassformer.py       # RadarAttention + GlassSegFormerRGBRadar
│   ├── data/dataset.py             # GlassSegDataset + get_loaders
│   ├── losses.py                   # BCE + Lovasz hinge loss, IoU metric (§IV-B.3)
│   ├── train.py                    # training entry point
│   └── evaluate.py                 # benchmark vs. baselines (Tables II/III)
├── ros2_ws/src/
│   └── acconeer_ros2_driver/       # ROS2 radar driver + real-time pipeline
│       └── acconeer_ros2_driver/
│           ├── acconeer_iq_node.py     # XM125 IQ driver → range profile
│           ├── radar_mask_node.py      # radar-guided mask generation (§IV-A)
│           └── glassformer_node.py     # real-time segmentation inference
├── assets/                         # figures
├── docs/                           # project page (GitHub Pages)
├── requirements.txt
└── pyproject.toml
```

## Installation

### ML side (training / evaluation)

```bash
git clone https://github.com/Suhani92/GlassFormer.git
cd GlassFormer
python -m venv .venv && source .venv/bin/activate
pip install -e .          # installs the `glassformer` package + requirements
```

### ROS2 side (radar driver + real-time inference)

Requires ROS2 (Humble or later) and the Intel RealSense ROS wrapper.

```bash
# 1. Intel RealSense ROS wrapper (not vendored here — install upstream):
sudo apt install ros-$ROS_DISTRO-realsense2-camera
#    or build from source: https://github.com/IntelRealSense/realsense-ros

# 2. Acconeer radar tooling:
pip install "acconeer-exptool[app]"

# 3. Build this workspace:
cd ros2_ws
colcon build --packages-select acconeer_ros2_driver
source install/setup.bash
```

## Usage

### Training

```bash
python -m glassformer.train --data-root /path/to/dataset --img-size 402
```

### Evaluation

```bash
python -m glassformer.evaluate \
    --data-root /path/to/test_split \
    --radar-ckpt checkpoints/best_glassformer.pt \
    --segformer-ckpt checkpoints/best_segformer_baseline.pt \
    --save-dir results/
```

### Real-time pipeline (ROS2)

```bash
# Radar driver + radar-guided mask generation
ros2 launch acconeer_ros2_driver glass_launch.py serial_port:=/dev/ttyUSB0

# Segmentation inference node (point it at your checkpoint)
ros2 run acconeer_ros2_driver glassformer_node \
    --ros-args -p model_path:=/path/to/best_glassformer.pt
```

The dataset is expected as:

```
<data_root>/{train,val,test}/
    images/       # RGB frames
    radar_mask/   # radar-derived binary priors
    gt_masks/     # ground-truth glass masks
```

## Dataset

The synchronized RGB-D-radar dataset (Acconeer XM125 + Intel RealSense D455,
rigidly mounted via a 3D-printed bracket) will be released here.
<!-- TODO: add download link -->

## Citation

```bibtex
@inproceedings{grover2026glassformer,
  title     = {GlassFormer: Learning Real-time Glass Segmentation using Radar-Depth Fusion},
  author    = {Grover, Suhani and Srivastava, Astik and Dinesh, Viswas and Sharma, Avinash and Krishna, Madhava},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

## Acknowledgement

This work was supported by IHub-Data via project M2-029.

## License

Released under the [MIT License](LICENSE).
