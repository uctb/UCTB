## Currently Supported Models

###  ARIMA (Autoregressive Integrated Moving Average)

ARIMA is a simple and widely used time series prediction model.

- Reference Paper:

  + [Williams, B. M., & Hoel, L. A. (2003). Modeling and forecasting vehicular traffic flow as a seasonal ARIMA process: Theoretical basis and empirical results](https://www3.nd.edu/~busiforc/handouts/ARIMA%20Engineering%20Article.pdf)
- Reference Package: `pandas`, `statsmodels`

###  DCRNN (Diffusion Convolutional Recurrent Neural Network)

DCRNN is a deep learning framework for traffic forecasting that incorporates both spatial and temporal dependency in the traffic flow. It captures the spatial dependency using bidirectional random walks on the graph, and the temporal dependency using the encoder-decoder architecture with scheduled sampling.

- Reference Paper:

  + [Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2017). Diffusion convolutional recurrent neural network: Data-driven traffic forecasting](https://arxiv.org/abs/1707.01926)

###  DeepST (Deep learning-based prediction model for Spatial-Temporal data)

DeepST is composed of three components: 1) temporal dependent instances: describing temporal closeness, period and seasonal trend; 2) convolutional neural networks: capturing near and far spatial dependencies; 3) early and late fusions: fusing similar and different domains' data.

- Reference Paper:

  + [Zhang, J., Zheng, Y., Qi, D., Li, R., & Yi, X. (2016, October). DNN-based prediction model for spatio-temporal data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/09/DeepST-SIGSPATIAL2016.pdf)

###  GeoMAN (Multi-level Attention Networks for Geo-sensory Time Series Prediction)

GeoMAN consists of two major parts: 1) A multi-level attention mechanism (including both local and global  spatial attentions in encoder and temporal attention in decoder) to model the dynamic spatio-temporal  dependencies; 2) A general fusion module to incorporate the external factors from different domains (e.g.,  meteorology, time of day and land use).

- Reference Paper:

  + [Liang, Y., Ke, S., Zhang, J., Yi, X., & Zheng, Y. (2018, July). GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction](https://www.ijcai.org/proceedings/2018/0476.pdf)

###  HM (Historical Mean)

HM is a constant model and always forecasts the sample mean of the historical data.

###  HMM (Hidden Markov Model)

Hidden Markov Model is a statistical Markov model in which the system being modeled is assumed to be a Markov process with hidden states. It is often used in temporal pattern recognition.

- Reference Paper:

  + [Chen, Z., Wen, J., & Geng, Y. (2016, November). Predicting future traffic using hidden markov models](https://ieeexplore.ieee.org/abstract/document/7785328)
- Reference Package: `hmmlearn`

###  ST-MGCN (Spatiotemporal Multi-graph Convolution Network)

ST-MGCN is a deep learning based model which encoded the non-Euclidean correlations among regions using multiple graphs and explicitly captured them using multi-graph convolution.

- Reference Paper:

  + [Geng, X., Li, Y., Wang, L., Zhang, L., Yang, Q., Ye, J., & Liu, Y. (2019). Spatiotemporal multi-graph convolution network for ride-hailing demand forecasting](https://ieeexplore.ieee.org/abstract/document/7785328)


### ST-ResNet

ST-ResNet is a deep-learning model with an end-to-end structure based on unique properties of spatio-temporal data making use of convolution and residual units.

- Reference Paper:
  - [Zhang, J., Zheng, Y., & Qi, D. (2017, February). Deep spatio-temporal residual networks for citywide crowd flows prediction](https://arxiv.org/pdf/1610.00081.pdf)

### STMeta

STMeta is our prediction model, which requires extra graph information as input, and combines Graph Convolution LSTM and Attention mechanism. 

- Reference Package: `tensorflow`

### XGBoost

XGBoost is a gradient boosting machine learning algorithm widely used in flow prediction and other machine learning prediction areas.

- Reference Paper:
  - [Alajali, W., Zhou, W., Wen, S., & Wang, Y. (2018). Intersection Traffic Prediction Using Decision Tree Models](https://www.mdpi.com/2073-8994/10/9/386)
- Reference Package: `xgboost`

------

<u>[Back To HomePage](../../index.html)</u>

