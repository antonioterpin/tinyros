"""Setup script for the TinyROS package."""

import pathlib

import setuptools

setuptools.setup(
    name="tinyros",
    version="0.1.0",
    author="Antonio Terpin",
    author_email="aterpin@ethz.ch",
    description="TinyROS: A lightweigth ROS implementation for GPU-first robots.",
    url="http://github.com/antonioterpin/tinyros",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=["tinyros"],
    include_package_data=True,
    install_requires=[
        # "jax[cuda12]==0.5.3",
        "setuptools",
        "cupy-cuda12x",
        "pytest",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
