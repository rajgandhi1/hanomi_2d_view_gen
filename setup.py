#!/usr/bin/env python3
"""
Setup script for the 3D to 2D CAD Drawing View Optimizer

This script helps users set up the environment and dependencies needed
to run the CAD analysis solution.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True


def check_conda():
    """Check if conda is available"""
    try:
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✓ Conda available: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("! Conda not found - pythonocc-core installation may be difficult")
        print("  Consider installing Miniconda or Anaconda")
        return False


def install_pythonocc():
    """Install pythonocc-core via conda"""
    print("\nInstalling pythonocc-core...")
    try:
        cmd = ['conda', 'install', '-y', 'pythonocc-core=7.7.2', '-c', 'conda-forge']
        subprocess.run(cmd, check=True)
        print("✓ pythonocc-core installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install pythonocc-core: {e}")
        print("You may need to install it manually:")
        print("  conda install pythonocc-core=7.7.2 -c conda-forge")
        return False


def install_pip_dependencies():
    """Install dependencies via pip"""
    print("\nInstalling Python dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
        subprocess.run(cmd, check=True)
        print("✓ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def verify_installation():
    """Verify that all required packages can be imported"""
    print("\nVerifying installation...")
    
    packages = [
        ('numpy', 'NumPy'),
        ('open3d', 'Open3D'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow')
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - Failed to import")
            all_good = False
    
    # Check pythonocc-core separately
    try:
        from OCC.Core import STEPControl_Reader
        print("✓ pythonocc-core")
    except ImportError:
        print("✗ pythonocc-core - Failed to import")
        all_good = False
    
    return all_good


def create_sample_files():
    """Create sample files for testing"""
    print("\nCreating sample files...")
    
    # Create a simple critical faces file
    sample_faces = {
        "critical_faces": [0, 1, 2, 3, 4, 5],
        "part_name": "sample_box",
        "description": "All faces of a sample box for testing"
    }
    
    import json
    with open("sample_critical_faces.json", 'w') as f:
        json.dump(sample_faces, f, indent=2)
    
    print("✓ Created sample_critical_faces.json")
    
    # Create run script
    run_script = """#!/bin/bash
# Sample run script for 3D to 2D CAD Drawing View Optimizer

echo "3D to 2D CAD Drawing View Optimizer"
echo "=================================="

# Check if STEP file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <step_file> [critical_faces.json] [output_dir]"
    echo ""
    echo "Examples:"
    echo "  $0 part.step"
    echo "  $0 part.step my_critical_faces.json"
    echo "  $0 part.step my_critical_faces.json my_output/"
    exit 1
fi

STEP_FILE=$1
CRITICAL_FACES=${2:-"sample_critical_faces.json"}
OUTPUT_DIR=${3:-"output"}

echo "STEP file: $STEP_FILE"
echo "Critical faces: $CRITICAL_FACES"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the analyzer
python main.py "$STEP_FILE" "$CRITICAL_FACES" "$OUTPUT_DIR"
"""
    
    with open("run_analysis.sh", 'w') as f:
        f.write(run_script)
    
    # Make executable on Unix-like systems
    if os.name != 'nt':
        os.chmod("run_analysis.sh", 0o755)
    
    print("✓ Created run_analysis.sh")


def main():
    """Main setup function"""
    print("3D to 2D CAD Drawing View Optimizer - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check conda availability
    has_conda = check_conda()
    
    if has_conda:
        print("\n" + "=" * 40)
        print("RECOMMENDED: Use conda environment")
        print("=" * 40)
        print("For the best experience, create a conda environment:")
        print("  conda env create -f environment.yml")
        print("  conda activate 3d-cad-optimizer")
        print("")
        
        response = input("Do you want to continue with manual setup instead? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Please use the conda environment setup and run the program again.")
            return True
    
    # Install dependencies
    if has_conda:
        print("\n" + "=" * 30)
        print("Installing dependencies...")
        print("=" * 30)
        
        # Ask user if they want to install pythonocc-core
        response = input("\nInstall pythonocc-core via conda? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            install_pythonocc()
    
    # Install pip dependencies
    print("\n" + "=" * 30)
    print("Installing Python packages...")
    print("=" * 30)
    install_pip_dependencies()
    
    # Verify installation
    print("\n" + "=" * 30)
    print("Verifying installation...")
    print("=" * 30)
    if verify_installation():
        print("\n✓ All packages installed successfully!")
    else:
        print("\n✗ Some packages failed to install")
        print("Please check error messages above and install missing packages manually")
        return False
    
    # Create sample files
    print("\n" + "=" * 30)
    print("Creating sample files...")
    print("=" * 30)
    create_sample_files()
    
    # Final instructions
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("IMPORTANT: If you haven't already, use the conda environment:")
    print("  conda env create -f environment.yml")
    print("  conda activate 3d-cad-optimizer")
    print("")
    print("To test the installation:")
    print("  python test_example.py")
    print("")
    print("To run with your own STEP file:")
    print("  python main.py your_part.step sample_critical_faces.json output/")
    print("")
    print("To generate custom critical faces:")
    print("  python generate_sample_faces.py -n 5 -o my_faces.json")
    print("")
    print("For help:")
    print("  python main.py --help")
    print("  python generate_sample_faces.py --help")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 