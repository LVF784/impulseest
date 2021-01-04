import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="impulseestest", # Replace with your own username
    version="0.0.11",
    author="Luan V. Fiorio",
    author_email="vfluan@gmail.com",
    description="Nonparametric impulse response estimation",
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
