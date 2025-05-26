
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="machine-failure-prediction",
    version="1.0.0",
    author="Your Name", # <<-- IMPORTANT: Change this to your name
    author_email="your.email@example.com", # <<-- IMPORTANT: Change this to your email
    description="A comprehensive machine learning system for predicting machine failures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/machine-failure-prediction", # <<-- IMPORTANT: Change this to your GitHub URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "machine-failure-prediction=main:main",
        ],
    },
)