## Installation

### Install UCTB

##### Step 1: Install TensorFlow

You can skip to step 2 if you already installed tensorflow.

You can refer to this page <https://www.tensorflow.org/install> to install tensorflow, if you have a Nvidia GPU installed on you computer, we highly recommend you to install GPU version of tensorflow.

##### Step 2: Install UCTB

```bash
pip install --upgrade UCTB
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