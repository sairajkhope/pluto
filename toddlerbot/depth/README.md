# ToddlerBot Depth Estimation Module

This module provides stereo depth estimation using FoundationStereo with TensorRT acceleration for real-time performance on Jetson Orin.

## Overview

The depth module uses [FoundationStereo](https://github.com/NVlabs/FoundationStereo) optimized with TensorRT for efficient inference on edge devices. It processes stereo camera pairs to generate accurate disparity maps.

## Setup

### Prerequisites

1. **Follow Jetson Orin Setup Instructions**

   First, complete the Jetson Orin setup by following the [official ToddlerBot documentation](https://hshi74.github.io/toddlerbot/software/02_jetson_orin.html). Pay special attention to the **stereo depth estimation** section.

2. **Run Automated Setup Script**

   Install all depth estimation dependencies automatically:
   ```bash
   bash scripts/setup_depth_module.sh
   ```

   This script will:
   - Install PyCUDA and PyTools
   - Configure system TensorRT access
   - Install cuSPARSELt libraries
   - Install PyTorch for Jetson
   - Build and install ONNX-TensorRT

3. **TensorRT Engine**

   **Option A: Use Prebuilt Engine**

   We provide prebuilt wheels for Jetson Orin NX 16GB with JetPack 6.1.

   Download the prebuilt TensorRT engine file:
   - [foundation_stereo_vits_96x128_16.engine](https://drive.google.com/drive/folders/1lha_uut-M5f63L8MkBZnnlF-yQJe0VD_?usp=sharing)

   **Engine Naming Convention**: `foundation_stereo_vits_{height}x{width}_{valid_iters}.engine`

   Place the engine file in:
   ```bash
   toddlerbot/depth/models/foundation_stereo_vits_96x128_16.engine
   ```

   **Option B: Build Engine Yourself**

   If you need a different configuration or are using a different JetPack version, follow the [FoundationStereo ONNX/TensorRT instructions](https://github.com/NVlabs/FoundationStereo/?tab=readme-ov-file#onnxtensorrt-inference-experimental).

3. **Calibration and Rectification Parameters**

   The system requires calibration and rectification parameters for your stereo camera setup. Generate these using:

   ```bash
   # Collect calibration images (place a 9x6 checkerboard in view)
   python examples/test_calibrate.py --collect

   # Perform calibration and rectification
   python examples/test_calibrate.py --calibrate --rectify
   ```

   This will create:
   - `toddlerbot/depth/params/calibration.pkl`
   - `toddlerbot/depth/params/rectification.npz`

## Usage

### Run Depth Estimation

```bash
# Run with visualization and save depth maps and point clouds
python examples/test_foundation_stereo.py --vis --save-output --engine <engine_path> --calib_params <calib_params_path> --rec_params <rec_params_path>
```


### Debug Mode

For testing without cameras connected:
```bash
# Place test images in results/depth_blake_debug/
# Format: {frame}_original_left.jpg, {frame}_original_right.jpg
python examples/test_foundation_stereo.py --debug results/depth_blake_debug --vis --save-output
```

## Performance

On Jetson Orin NX 16GB with JetPack 6.1:
- Input resolution: 96x128 (downsized from 640x480)
- Inference time: ~30-50ms per frame
- Full pipeline: ~60-80ms (including rectification and point cloud generation)

## Comparing Depth Results

To evaluate depth estimation accuracy:

```bash
# Compare depth maps and point clouds between two runs
python toddlerbot/visualization/vis_depth_comparison.py \
    --gt-folder results/depth_blake_0529_fs_480x640_object \
    --pred-folder results/depth_blake_0529_fs_96x128_object \
    --frame 10 \
```

This generates:
- Depth error heatmaps
- Point cloud alignment metrics
- Bird's eye view overlays

## Troubleshooting

1. **ImportError for tensorrt/pycuda**: Ensure you've followed the JetPack 6.1 setup instructions and configured TensorRT paths correctly.

2. **Calibration file not found**: Run the calibration procedure or update the paths in your code to point to your calibration files.

3. **Engine file incompatible**: TensorRT engines are specific to GPU architecture and TensorRT version. Rebuild the engine if switching hardware or JetPack versions.

4. **Low FPS**: Try reducing `--valid_iters` and image size when building the TensorRT engine files.

## Citation

If you use this module in your research, please cite:

```bibtex
@article{wen2025foundationstereo,
  title={FoundationStereo: Zero-Shot Stereo Matching},
  author={Wen, Bowen and others},
  journal={CVPR},
  year={2025}
}
```
