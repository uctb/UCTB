## Currently Supported Models

###  Historical Mean

Currently Historical Mean method predicts the flow at a certain time slot according to the historical mean value of a specific number of nearest previous time slots.

###  ARIMA (Autoregressive Integrated Moving Average)

ARIMA is a simple and widely used time series prediction model.

- Reference Paper:

  + [Williams, B. M., & Hoel, L. A. (2003). Modeling and forecasting vehicular traffic flow as a seasonal ARIMA process: Theoretical basis and empirical results](https://www3.nd.edu/~busiforc/handouts/ARIMA%20Engineering%20Article.pdf)

- Reference Package: `pandas`, `statsmodels`

###  HMM (Hidden Markov Model)

Hidden Markov Model is a statistical Markov model in which the system being modeled is assumed to be a Markov process with hidden states. It is often used in temporal pattern recognition.

- Reference Paper:

  + [Chen, Z., Wen, J., & Geng, Y. (2016, November). Predicting future traffic using hidden markov models](https://ieeexplore.ieee.org/abstract/document/7785328)

- Reference Package: `hmmlearn`

### XGBoost

XGBoost is a gradient boosting machine learning algorithm widely used in flow prediction and other machine learning prediction areas.

- Reference Paper:
  - [Alajali, W., Zhou, W., Wen, S., & Wang, Y. (2018). Intersection Traffic Prediction Using Decision Tree Models](https://www.mdpi.com/2073-8994/10/9/386)
- Reference Package: `xgboost`

### AMulti-GCLSTM

AMulti-GCLSTM (Attention-Fused Multi-Graph Convolutional LSTM Networks) is our prediction model, which requires extra graph information as input, and combines Graph Convolution LSTM and Attention mechanism. 

- Reference Package: `tensorflow`

------

<u>[Back To HomePage](../index.html)</u>

