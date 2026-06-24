from setuptools import setup, find_packages
setup(
    name="rag_chunking_bench",
    version="1.0.0",
    description="Reproduction of arXiv:2606.00881 — RAG Chunking Methods Evaluation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
