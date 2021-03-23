import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="impulseest",
    version="1.0",
    author="Luan V. Fiorio",
    author_email="vfluan@gmail.com",
    description="non-parametric impulse response estimation with input-output data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LVF784/impulseest",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)