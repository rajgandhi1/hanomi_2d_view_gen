# 3D to 2D CAD Drawing View Optimizer

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended for pythonocc-core installation)

### Setup

**Option 1: Using conda environment (Recommended)**
```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate 3d-cad-optimizer
```

**Option 2: Manual installation**
```bash
# Install pythonocc-core via conda
conda install pythonocc-core=7.7.2 -c conda-forge

# Install other dependencies via pip
pip install -r requirements.txt
```

### Dependencies
- `pythonocc-core==7.7.2`: STEP file loading and B-Rep operations
- `open3d>=0.19.0`: Mesh operations and ray-casting
- `numpy>=1.21.0`: Numerical computations
- `matplotlib>=3.5.0`: Visualization utilities
- `Pillow>=9.0.0`: Image processing 