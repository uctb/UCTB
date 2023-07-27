import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="UCTB",
    version="0.3.5",
    author="UCTB group",
    author_email="chenliyue2019@gmail.com",
    description="Urban Computing ToolBox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uctb/UCTB",
    packages=setuptools.find_packages(),
    install_requires=['hmmlearn>=0.2.1',
                      'keras==2.2.4',
                      'GPUtil',
                      'numpy>=1.16.2',
                      'pandas>=0.24.2',
                      'python-dateutil',
                      'scikit-learn>=0.20.3',
                      'scipy==1.10.0',
                      'statsmodels>=0.9.0',
                      'wget>=3.2',
                      'xgboost>=0.82',
                      'nni>=0.8',
                      'chinesecalendar>=1.2.2',
                      'workalendar>=8.2.0',
                      'PyYAML>=5.1.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)