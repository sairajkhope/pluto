#!/bin/bash

# ToddlerBot Depth Estimation Setup Script for Jetson Orin with JetPack 6.1
#
# This script automates the installation steps detailed in:
# toddlerbot/depth/README.md
#
# Prerequisites:
# - Conda environment already set up (see https://hshi74.github.io/toddlerbot/software/01_setup.html#set-up-conda-environment)
# - JetPack 6.1 installed on Jetson device
# - This script should be run from within the activated toddlerbot conda environment

echo "=========================================="
echo "ToddlerBot Depth Estimation Setup Script"
echo "=========================================="
echo ""
echo "This script will install dependencies for depth estimation on Jetson Orin."
echo "For detailed step-by-step instructions, see: toddlerbot/depth/README.md"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install conda first following: https://hshi74.github.io/toddlerbot/software/01_setup.html#set-up-conda-environment"
    exit 1
fi

# Check if we're in a conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "ERROR: No conda environment is active"
    echo "Please activate the toddlerbot conda environment: conda activate toddlerbot"
    exit 1
fi

echo "Passed:Conda is available (active environment: ${CONDA_DEFAULT_ENV})"

# Check for JetPack 6.1
if [[ ! -d "/usr/local/cuda-12.6" ]]; then
    echo "WARNING: CUDA 12.6 directory not found"
    echo "This script is designed for JetPack 6.1 which includes CUDA 12.6"
fi

echo ""
read -p "Have you completed the JetPack 6.1 setup as described in the ToddlerBot documentation? (y/N): " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please complete JetPack 6.1 setup first:"
    echo "https://hshi74.github.io/toddlerbot/software/02_jetson_orin.html"
    exit 1
fi

echo ""
read -p "Are you ready to proceed with automatic dependency installation? (y/N): " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled. Run this script again when ready."
    exit 0
fi

echo ""
echo "Setting up depth estimation dependencies for ToddlerBot on Jetson Orin..."

# Set CUDA paths for JetPack 6.1 (includes CUDA 12.6)
# These environment variables are needed for PyCUDA compilation
echo "Configuring CUDA environment variables..."
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_ROOT=/usr/local/cuda
echo "Installing PyCUDA..."
pip install pycuda

# Download cuSPARSELt package for Jetson (aarch64) with Ubuntu 22.04
# URL from: https://developer.nvidia.com/cusparselt-downloads
echo "Installing cuSPARSELt for optimized sparse operations..."
wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb

# Install the downloaded package
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/

# Update package lists and install cuSPARSELt libraries
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev

# Download PyTorch wheel from NVIDIA's Jetson PyTorch repository
# URL from: https://developer.nvidia.com/embedded/downloads (search "PyTorch for Jetson")
wget --content-disposition https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Create a .pth file to make system TensorRT installation accessible to conda environment
# This allows the conda environment to use the TensorRT installed with JetPack
echo "Configuring TensorRT Python bindings..."
echo "/usr/lib/python3.10/dist-packages" \
    > $CONDA_PREFIX/lib/python3.10/site-packages/tensorrt_global.pth

echo "Installing ONNX-TensorRT..."
# Clone and build onnx-tensorrt
git clone --recursive https://github.com/onnx/onnx-tensorrt.git
cd onnx-tensorrt
git checkout release/10.13-GA
python -c "import setuptools; setuptools.setup(name='onnx_tensorrt', version='10.13.0', packages=['onnx_tensorrt'], install_requires=['pycuda', 'numpy', 'onnx'])" bdist_egg

# Install the egg
cp dist/onnx_tensorrt-10.13.0-py3.10.egg $CONDA_PREFIX/lib/python3.10/site-packages/
echo "$CONDA_PREFIX/lib/python3.10/site-packages/onnx_tensorrt-10.13.0-py3.10.egg" > $CONDA_PREFIX/lib/python3.10/site-packages/onnx_tensorrt.pth

# Verify installation
pip list | grep onnx_tensorrt

# Go back and clean up
cd ..
rm -rf onnx-tensorrt

echo "Cleaning up downloaded files..."
rm *.whl
rm *.deb

echo ""
echo "=========================================="
echo "Depth estimation setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download the TensorRT engine file from:"
echo "   https://drive.google.com/drive/folders/1lha_uut-M5f63L8MkBZnnlF-yQJe0VD_?usp=sharing"
echo "2. Place it in: toddlerbot/depth/models/"
echo "3. See toddlerbot/depth/README.md for usage instructions"