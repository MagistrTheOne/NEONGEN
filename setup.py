"""
Setup script for QTG (Quantum-Topological-Geometric) Model
"""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name="qtg-model",
    version="1.0.0",
    author="MagistrTheOne",
    author_email="",
    description="Quantum-Topological-Geometric Fusion Model for Text Generation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/MagistrTheOne/qtg-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="quantum topological geometric ai nlp transformer",
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.5.0",
        ],
        "gpu": [
            "torch>=2.0.1+cu118",
            "torchvision>=0.15.2+cu118",
        ],
        "topology": [
            "gudhi>=3.8.0",
            "ripser>=0.6.4",
            "persim>=0.3.2",
        ],
        "quantum": [
            "pennylane>=0.32.0",
            "qiskit>=0.44.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qtg-train=train:main",
            "qtg-generate=src.api.generate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
