"""
Package setup for ml-pipeline-monitor.
"""
from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

setup(
    name="ml-pipeline-monitor",
    version="1.0.0",
    author="Manpat Ell",
    description="Production-grade MLOps observability platform built with Streamlit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manpatell/ml-pipeline-monitor",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
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
            "mlmonitor=src.cli:main",
            "mlmonitor-api=services.api.main:run",
        ],
    },
)
