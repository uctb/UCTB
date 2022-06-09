## 技术框架

本研究基于交通轨迹数据、城市环境数据、事件相关信息等多源异构空间群智大数据，在使用GCN-GRU对群体流量进行时空建模的基础上，开创性地从时间和空间两方面对事件的影响进行建模，时间上我们基于Multi-Task结合事件的时序特征与区域流量进行协同训练，空间上基于刺激响应建模时间对于区域流量的影响，进一步基于Multi-View将事件的空间影响融合到区域流量中。

![研究框架](https://research.crowdsensing.cn/algorithm-platform/event-graph/images/event_graph_framework.png)

## 代码修改

### 新增`Experiments\STORM\DirRec_STORM.py`

### 新增`model\STORM.py`

### 修改`model\__init__.py`

新增一行代码：

```python
from .STORM import STORM
```

### 修改`preprocess\GraphGenerator.py`

增加基于 `function` （功能区）的建图，即根据不同区域的 `POI` 相似度来作为区域间的边。

```python
    def build_graph(self, graph_name):
        if graph_name.lower() == 'function':
            AM = self.function_adjacent(self.dataset.data['Node']['POI'], threshold=float(self.threshold_correlation))
            LM = self.adjacent_to_laplacian(AM)
            
    @staticmethod
    def function_adjacent(poi_data, threshold):
        '''
        Calculate function graph based on pearson coefficient.

        Args:
            poi_data(ndarray): numpy array with shape [num_node, poi_type_num].
            threshold(float): float between [-1, 1], nodes with Pearson Correlation coefficient
                larger than this threshold will be linked together.
        '''
        adjacent_matrix = np.zeros([poi_data.shape[0], poi_data.shape[0]])
        for i in range(poi_data.shape[0]):
            for j in range(poi_data.shape[0]):
                r, p_value = pearsonr(poi_data[i, :], poi_data[j, :])
                adjacent_matrix[i, j] = 0 if np.isnan(r) else r
        adjacent_matrix = (adjacent_matrix >= threshold).astype(np.float32)
        return adjacent_matrix
```

### 修改`model_unit/GraphModelLayers.py`

由于需要对多个图做图卷积，原始`GCL`方法在对第二个图做图卷积时会报下面这个错误：

> ValueError: Variable multi_gcl/gcl_0/weights already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:

因此修改该文件中第 `178~183` 行修改为如下所示：

```python
    @staticmethod
    def add_multi_gc_layers(inputs, graph_id, gcn_k, gcn_l, output_size, laplacian_matrix, activation=tf.nn.tanh):
        '''
        Call add_gc_layer function to add multi Graph Convolution Layer.`gcn_l` is the number of layers added.
        '''
        with tf.variable_scope('multi_gcl_%s' % graph_id, reuse=False):
```

## 运行方法

```shell
nohup python -u DirRec_STORM.py >DirRec_STORM.log 2>&1 &
```
