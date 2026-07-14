from setuptools import setup, find_packages

setup(
    name="dcnp",
    version="1.0.0",
    description="Replication: Dissecting Characteristics Nonparametrically (Freyberger, Neuhierl & Weber 2017)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "statsmodels>=0.14.0",
        "pyyaml>=6.0",
    ],
)
