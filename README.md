# UCTB (Urban Computing Tool Box)

 [![Python](https://img.shields.io/badge/python-3.6%7C3.7-blue)]() [![PyPI](https://img.shields.io/badge/pypi%20package-v0.3.0-sucess)](https://pypi.org/project/UCTB/) ![tensorflow](https://img.shields.io/badge/tensorflow-1.13-important)[![https://img.shields.io/badge/license-MIT-green](https://img.shields.io/badge/license-MIT-green)]() 

------

### News

**2021-11**: Our paper on UCTB, entitled '*Exploring the Generalizability of Spatio-Temporal Traffic Prediction: Meta-Modeling and an Analytic Framework*', has been accepted by IEEE TKDE! [[IEEE Xplore](https://ieeexplore.ieee.org/document/9627543)][[arXiv](https://arxiv.org/abs/2009.09379)]

------

**Urban Computing Tool Box** is a package providing **spatial-temporal prediction models** for various urban computing tasks, such as traffic prediction, crowd flow prediction, ridesharing demand prediction, etc. It contains both conventional models and state-of-art models. 

Currently the package supported the following models: ([Details](https://uctb.github.io/UCTB/md_file/static/current_supported_models.html))

- ARIMA
- DCRNN
- DeepST
- GeoMAN
- AGCRN
- ASTGCN
- GraphWaveNet
- GMAN
- STSGCN
- STGCN
- HM
- HMM
- ST-MGCN
- ST-ResNet
- STMeta
- XGBoost

UCTB is a flexible and open package. You can use the data we provided or use your own data, the data structure is well stated in the tutorial chapter. You can build your own model based on model-units we provided and use the model-training class to train the model.

UCTB toolbox may not work successfully with the upgrade of some packages. We thus encourage you to use the specific version of packages or use our docker environment to avoid these unseen errors.

```
python==3.6
tensorflow==1.13
Keras==2.2.4
h5py==2.9.0
```

[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://uctb.github.io/UCTB)

