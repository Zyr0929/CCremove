from setuptools import setup, find_packages

setup(
    name="subclean",
    version="1.0.0",
    description="Zero-blur deep generative video caption and subtitle removal pipeline.",
    author="Deep Learning Computer Vision Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "pillow>=9.5.0",
        "scikit-image>=0.20.0",
        "easyocr>=1.7.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.8",
        "gradio>=4.20.0",
        "psutil>=5.9.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
        "einops>=0.6.1",
    ],
    entry_points={
        "console_scripts": [
            "subclean=run:main",
        ],
    },
)
