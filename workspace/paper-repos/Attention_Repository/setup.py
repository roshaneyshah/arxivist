from setuptools import setup, find_packages

setup(
    name="transformer-aiayn",
    version="1.0.0",
    description="Reproducible implementation of 'Attention Is All You Need' (Vaswani et al., 2017)",
    author="ArXivist v1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "sentencepiece>=0.1.99",
        "sacrebleu>=2.3.1",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
    ],
)
