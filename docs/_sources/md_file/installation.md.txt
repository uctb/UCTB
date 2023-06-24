## Installation

UCTB is based on several well-known deep learning frameworks, including PyTorch, TensorFlow, and MXNet. If you have an Nvidia GPU installed on your computer, we highly recommend you install the GPU version of these frameworks.

UCTB toolbox may not work successfully with the upgrade of some packages. We thus encourage you to use the specific version of packages to avoid unseen errors. ***To avoid potential conflict, we highly recommend you install UCTB vis Anaconda or use our docker environment.***

```bash
# Install PyTorch
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# Install TensorFlow
conda install tensorflow-gpu==1.13.1

# Install MXNet
pip install mxnet-cu100==1.5.0
```

### Install via Anaconda

**Step 1: Install Anaconda**

You can skip to step 2 if you already installed Anaconda.

You can refer to this page <https://www.anaconda.com/download> to install Anaconda.

**Step 2: create UCTB environment**

Create the UCTB environment by the following command. You may need the [environment.yaml](https://github.com/uctb/UCTB/blob/master/environment.yaml) file.

```
conda env create -f environment.yaml
```

Then activate the UCTB enviroment.

```
conda activate UCTB
```

**Step 3: Install UCTB**

```bash
pip install --upgrade UCTB
```

Then we finish installing UCTB, and we could start it in the conda environment.



**[Optional Step] If fail in Step 3**

If you fail to install UCTB via the pip command (This situation may occur when your OS is Windows), you could install UCTB by source code.

First, download UCTB's source code, then go to the main folder of UCTB. Your directory may look like

```bash
XXX/UCTB# ls
build_install.py  dist  environment.yaml  __init__.py  QuickStarts  setup.py
build.py          docs  Experiments       LICENSE      README.md    UCTB
```

Then run the following command.

```bash
python build_install.py 
```

The following required package will be installed or upgraded with UCTB:

```bash
'hmmlearn',
'keras',
'GPUtil',
'numpy',
'pandas',
'python-dateutil',
'scikit-learn',
'scipy',
'statsmodels',
'wget',
'xgboost',
'nni',
'chinesecalendar',
'PyYAML'
```

### UCTB Docker

You can also  use UCTB by docker. First pull uctb docker from docker hub.

```bash
docker pull dichai/uctb:v0.2.0
```

And  you then can run it.

```bash
docker run  --runtime=nvidia  -it -d dichai/uctb:v0.2.0 /bin/bash
```

<u>[Back To HomePage](../index.html)</u>