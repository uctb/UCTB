## Currently Supported Models

### AGCRN (Adaptive Graph Convolutional Recurrent Network)

AGCRN is a deep nerual network for traffic prediction consisting of two adaptive module and recurrent networks.

- Reference Paper:
  - [Bai, L., Yao, L., Li, C., Wang, X., & Wang, C. (2020). Adaptive graph convolutional recurrent network for traffic forecasting.](https://proceedings.neurips.cc/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf)
- Reference Implementation:
  - [Github repository (LeiBAI)](https://github.com/LeiBAI/AGCRN)

###  ARIMA (Autoregressive Integrated Moving Average)

ARIMA is a widely used classical statistical model on time series prediction.

- Reference Paper:

  + [Williams, B. M., & Hoel, L. A. (2003). Modeling and forecasting vehicular traffic flow as a seasonal ARIMA process: Theoretical basis and empirical results](https://www3.nd.edu/~busiforc/handouts/ARIMA%20Engineering%20Article.pdf)
- Reference Package: `pandas`, `statsmodels`

### ASTGCN (Attenion Based Spatial-temporal Graph Convolutional Networks)

ASTGCN is a deep neural network for traffic flow forecasting. It models temporal-dependencies from three perspectives using attetion mechanism. And it models spatial-dependencies employing graph convolutions.

- Reference Paper:
  - [Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019, July). Attention based spatial-temporal graph convolutional networks for traffic flow forecasting.](https://ojs.aaai.org/index.php/AAAI/article/view/3881)
- Reference Implementation:
  - [Github repository (guoshnBJTU)](https://github.com/guoshnBJTU/ASTGCN-r-pytorch)

###  DCRNN (Diffusion Convolutional Recurrent Neural Network)

DCRNN is a deep learning framework for traffic forecasting that incorporates both spatial and temporal dependency in the traffic flow. It captures the spatial dependency using bidirectional random walks on the graph, and the temporal dependency using the encoder-decoder architecture with scheduled sampling.

- Reference Paper:

  + [Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2017). Diffusion convolutional recurrent neural network: Data-driven traffic forecasting](https://arxiv.org/abs/1707.01926)
- Reference Implementation: 
  + [A TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network (liyaguang)](https://github.com/liyaguang/DCRNN)

###  DeepST (Deep learning-based prediction model for Spatial-Temporal data)

DeepST is composed of three components: 1) temporal dependent instances: describing temporal closeness, period and seasonal trend; 2) convolutional neural networks: capturing near and far spatial dependencies; 3) early and late fusions: fusing similar and different domains' data.

- Reference Paper:

  + [Zhang, J., Zheng, Y., Qi, D., Li, R., & Yi, X. (2016, October). DNN-based prediction model for spatio-temporal data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/09/DeepST-SIGSPATIAL2016.pdf)

###  GeoMAN (Multi-level Attention Networks for Geo-sensory Time Series Prediction)

GeoMAN consists of two major parts: 1) A multi-level attention mechanism (including both local and global  spatial attentions in encoder and temporal attention in decoder) to model the dynamic spatio-temporal  dependencies; 2) A general fusion module to incorporate the external factors from different domains (e.g.,  meteorology, time of day and land use).

- Reference Paper:

  + [Liang, Y., Ke, S., Zhang, J., Yi, X., & Zheng, Y. (2018, July). GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction](https://www.ijcai.org/proceedings/2018/0476.pdf)
- Reference Implementation:
  + [An easy implement of GeoMAN using TensorFlow (yoshall & CastleLiang)](https://github.com/yoshall/GeoMAN)

### GMAN (Graph Multi-Attention Network)

GMAN is a deep nerual network for traffic prediction adopting encoder-decoder architecture. Both encode and decoder consist of multiple spatio-temporal attention blocks to model spatio-temporal dependencies.

- Reference Paper:
  - [Zheng, C., Fan, X., Wang, C., & Qi, J. (2020, April). Gman: A graph multi-attention network for traffic prediction.](https://ojs.aaai.org/index.php/AAAI/article/view/5477)
- Reference Implementation:
  - [implementation of Graph Multi-Attention Network](https://github.com/zhengchuanpan/GMAN)

### GraphWaveNet

GraphWaveNet is an end-to-end novel graph neural network. It captures spatial dependencies through a self-adptive adjacency matrix. And it captures temporal dependencies through convolutions.

- Reference Paper:
  - [Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph wavenet for deep spatial-temporal graph modeling.](https://www.ijcai.org/proceedings/2019/0264.pdf)
- Reference Implementation:
  - [Github repository (VeritasYin)](https://github.com/VeritasYin/STGCN_IJCAI-18)

###  HM (Historical Mean)

HM is a constant model and always forecasts the sample mean of the historical data.

###  HMM (Hidden Markov Model)

Hidden Markov Model is a statistical Markov model in which the system being modeled is assumed to be a Markov process with hidden states. It is often used in temporal pattern recognition.

- Reference Paper:

  + [Chen, Z., Wen, J., & Geng, Y. (2016, November). Predicting future traffic using hidden markov models](https://ieeexplore.ieee.org/abstract/document/7785328)
- Reference Package: `hmmlearn`

### STGCN (Spatio-temporal Graph Convolutional Networks)

STGCN is a deep learning framework for traffic forecasting with complete convolutional structures.

- Reference Paper:
  - [Yu, B., Yin, H., & Zhu, Z. (2017). Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting.](https://www.ijcai.org/proceedings/2018/0505.pdf)
- Reference Implementation:
  - [Github repository (VeritasYin)](https://github.com/VeritasYin/STGCN_IJCAI-18)

### STMeta

STMeta is our prediction model, which requires extra graph information as input, and combines Graph Convolution LSTM and Attention mechanism.

- Reference Package: `tensorflow`

### ST-MGCN (Spatiotemporal Multi-graph Convolution Network)

ST-MGCN is a deep learning based model which encoded the non-Euclidean correlations among regions using multiple graphs and explicitly captured them using multi-graph convolution.

- Reference Paper:

  + [Geng, X., Li, Y., Wang, L., Zhang, L., Yang, Q., Ye, J., & Liu, Y. (2019). Spatiotemporal multi-graph convolution network for ride-hailing demand forecasting](https://ieeexplore.ieee.org/abstract/document/7785328)
- Reference Implementation:
  + [A PyTorch implementation of the ST-MGCN model  (shawnwang-tech)](https://github.com/shawnwang-tech/ST-MGCN-pytorch)


### ST-ResNet

ST-ResNet is a deep-learning model with an end-to-end structure based on unique properties of spatio-temporal data making use of convolution and residual units.

- Reference Paper:
  - [Zhang, J., Zheng, Y., & Qi, D. (2017, February). Deep spatio-temporal residual networks for citywide crowd flows prediction](https://arxiv.org/pdf/1610.00081.pdf)
- Reference Implementation:
  - [Github repository (lucktroy)](https://github.com/lucktroy/DeepST/tree/master/scripts/papers/AAAI17)

### STSGCN (Spatial-temporal Synchronous Graph Convolutional Networks)

STSGCN is a deep learning framework for spatial-temporal network data forecasting. It is able to capture spatial-temporal dependencies through a designed spatial-temporal synchronous modeling mechanism.

- Reference Paper:
  - [Song, C., Lin, Y., Guo, S., & Wan, H. (2020, April). Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting.](https://ojs.aaai.org/index.php/AAAI/article/view/5438)
- Reference Implementation:
  - [Github repository (Davidham3)](https://github.com/Davidham3/STSGCN)


### XGBoost

XGBoost is a gradient boosting machine learning algorithm widely used in flow prediction and other machine learning prediction areas.

- Reference Paper:
  - [Alajali, W., Zhou, W., Wen, S., & Wang, Y. (2018). Intersection Traffic Prediction Using Decision Tree Models](https://www.mdpi.com/2073-8994/10/9/386)
- Reference Package: `xgboost`

------

<u>[Back To HomePage](../../index.html)</u>
