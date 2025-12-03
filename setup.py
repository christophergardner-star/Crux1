from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cruxy",
    version="2.1.0",
    author="Axiom Forge Systems Ltd",
    author_email="dev@axiomforge.ai",
    description="The Cruxy Stability Engine: Adaptive Optimization for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/christophergardner-star/Crux1", 
    project_urls={
        "Bug Tracker": "https://github.com/christophergardner-star/Crux1/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib", "torchvision"],
    },
)
