from setuptools import setup, find_packages

setup(
    name="rl_trade_execution",
    version="1.0.0",
    description="RL Optimized Trade Execution — Nevmyvaka, Feng, Kearns (ICML 2006)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
    ],
)
