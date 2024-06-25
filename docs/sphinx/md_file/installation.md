## Installation

UCTB is based on several well-known deep learning frameworks, including **PyTorch**, **TensorFlow**, and **MXNet**. If you have an Nvidia GPU installed on your computer, we highly recommend you install the GPU version of these frameworks.

UCTB toolbox may not work successfully with the upgrade of some packages. We thus encourage you to use the specific version of packages to avoid unseen errors. ***To avoid potential conflict, we highly recommend you install UCTB vis Anaconda.***

### Install via Anaconda

**Step 1: Install Anaconda**

You can skip to step 2 if you already installed Anaconda.

To install Anaconda, please refer to this page <https://www.anaconda.com/download>.



**Step 2: create UCTB environment**

Create the UCTB environment by the following command. You may need the [environment.yaml](https://github.com/uctb/UCTB/blob/master/environment.yaml) file.

```bash
conda env create -f environment.yaml
```

Then activate the UCTB environment and start to use it. 🎉🎉🎉

```bash
conda activate UCTB
```



### Check for Success

If you  successfully install UCTB, you may get the following output after importing UCTB. 

```
(UCTB) XXX:~$ python
Type "help", "copyright", "credits" or "license" for more information.
>>> import UCTB
Using TensorFlow backend.
>>> 
```

### High version GPU Framework support

**Existing Problems**

Due to changes in the design architecture of high-version GPUs, low-version CUDA is not compatible with high-version GPUs. As a result, Tensorflow 1.x is only compatible with low-version CUDA, leading to runtime failures on machines equipped with high-version GPUs. https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible 

**Solution**

Thanks to the [Nvidia TensorFlow](https://github.com/NVIDIA/tensorflow) project, which was created to support newer hardware and improve libraries for NVIDIA GPU users using TensorFlow 1.x, we can now install UCTB on machines with newer GPUs. You can follow the installation tutorial below to start enjoying UCTB.

**Clone the project**

```sh
>>> git clone git@github.com:uctb/UCTB.git
>>> cd UCTB
```

**Create Anaconda Environment**

```sh
>>> conda create -n UCTB python=3.8
>>> conda activate UCTB
```

**Install Nvidia Tensorflow**

```sh
>>> pip install --user nvidia-pyindex
>>> pip install --user nvidia-tensorflow[horovod]
```

**Install UCTB from Source**

```sh
>>> python build_install.py
```



### Q & A

**Q: I fail to install PyTorch, TensorFlow, and MXNet, what is the version number of them?**

A: We recommend you install PyTorch==1.1.0, TensorFlow==1.13.1, and MXNet==1.5.0 with cuda version 10.0. We here give the installation command:

```bash
# Install PyTorch
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# Install TensorFlow
conda install tensorflow-gpu==1.13.1

# Install MXNet
pip install mxnet-cu100==1.5.0
```



**Q:  I'm using Windows OS, and my Anaconda reports that it cannot find the PyTorch 1.1.0 packages. How to install it?**

A: You could install it by the following command.

```bash
pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
```



**Q:  I fail to install UCTB via pip. How to install it ?**

A: This situation may occur when your OS is Windows. You could install UCTB by its source code. First download UCTB's source code and your folder may look like this:

```bash
XXX/UCTB-master# ls
build_install.py  dist  environment.yaml  __init__.py  QuickStarts  setup.py
build.py          docs  Experiments       LICENSE      README.md    UCTB
```

then build and install UCTB by the following command:

```bash
python build_install.py
```

<u>[Back To HomePage](../index.html)</u>