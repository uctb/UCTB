import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="UCTB",
    version="0.0.1",
    author="Di Chai, Leye Wang, Jin Xu",
    author_email="dchai@connect.ust.hk",
    description="Urban Computing ToolBox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Di-Chai/UCTB",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)