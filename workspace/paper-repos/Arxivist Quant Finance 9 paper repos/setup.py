from setuptools import setup, find_packages

setup(
    name="spotv2net",
    version="1.0.0",
    description="ArXivist-generated reproduction of SpotV2Net (arXiv:2401.06249)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
