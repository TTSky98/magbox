import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="magbox",
    version="0.1",
    author="Yutian Wang",
    author_email="857823501@qq.com",
    description="Simulate lattice magnetic model with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/abhi0610/pytorch-ssd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)