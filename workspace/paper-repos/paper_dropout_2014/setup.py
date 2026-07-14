"""
setup.py for the Dropout reproduction package.
Allows `pip install -e .` from the repo root.
"""
from setuptools import setup, find_packages

setup(
    name="dropout-repro",
    version="1.0.0",
    description="Reproduction of Srivastava et al. (2014) Dropout paper",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
    ],
)
