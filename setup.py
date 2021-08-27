# setup.py
# Setup installation for the application

from pathlib import Path

from setuptools import setup


BASE_DIR = Path(__file__).parent


# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]


dev_packages = [
    "black==21.5b1",
    "flake8==3.9.2",
    "isort==5.8.0",
    "mypy==0.812",
    "pre-commit==2.13.0",
]


setup(
    name="image-to-latex",
    version="0.1",
    license="MIT",
    description="Convert images to latex code.",
    author="King Yiu Suen",
    author_email="kingyiusuen@gmail.com",
    url="https://github.com/kingyiusuen/image-to-latex/",
    keywords=[
        "machine-learning",
        "deep-learning",
        "artificial-intelligence",
        "latex",
        "neural-network",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={
        "dev": dev_packages,
    },
)
