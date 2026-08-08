"""
Package setup for ml-pipeline-monitor.

This file is maintained for backward compatibility with older pip versions.
The canonical configuration is in pyproject.toml.
"""
from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

setup(
    name="ml-pipeline-monitor",
    version="1.0.0",
    author="Aryan41211",
    description="Production-grade MLOps observability platform built with Streamlit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aryan41211/ml-pipeline-monitor",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests*"]),
    python_requires=">=3.11",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "mlmonitor-api=ml_pipeline_monitor.api.__main__:run",
        ],
    },
)
