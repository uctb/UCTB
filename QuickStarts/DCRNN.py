import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import DCRNN
from UCTB.evaluation import metric


class my_data_loader(NodeTrafficLoader):

    def diffusion_matrix(self, filter_type='random_walk'):
        def calculate_random_walk_matrix(adjacent_mx):
            d = np.array(adjacent_mx.sum(1))
            d_inv = np.power(d, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = np.diag(d_inv)
            random_walk_mx = d_mat_inv.dot(adjacent_mx)
            return random_walk_mx
        assert len(self.AM) == 1

        diffusion_matrix = []
        if filter_type == "random_walk":
            diffusion_matrix.append(calculate_random_walk_matrix(self.AM[0]).T)
        elif filter_type == "dual_random_walk":
            diffusion_matrix.append(calculate_random_walk_matrix(self.AM[0]).T)
            diffusion_matrix.append(calculate_random_walk_matrix(self.AM[0].T).T)
        return np.array(diffusion_matrix, dtype=np.float32)


data_loader = my_data_loader(dataset='Metro', city='ShanghaiV1',
                             closeness_len=6, period_len=0, trend_len=0, graph='Distance', normalize=False)

diffusion_matrix = data_loader.diffusion_matrix()

batch_size = 64

DCRNN_Obj = DCRNN(num_nodes=data_loader.station_number,
                  num_diffusion_matrix=diffusion_matrix.shape[0],
                  num_rnn_units=64,
                  num_rnn_layers=1,
                  max_diffusion_step=2,
                  seq_len=data_loader.closeness_len,
                  use_curriculum_learning=False,
                  input_dim=1,
                  output_dim=1,
                  cl_decay_steps=1000,
                  target_len=1,
                  lr=1e-4,
                  epsilon=1e-3,
                  optimizer_name='Adam',
                  code_version='AMulti-QuickStart',
                  model_dir='model_dir',
                  gpu_device='0',
                  batch_size=batch_size)

# Build tf-graph
DCRNN_Obj.build()

print('Number of trainable parameters', DCRNN_Obj.trainable_vars)

# Training
DCRNN_Obj.fit(inputs=data_loader.train_closeness.transpose([0, 2, 1, 3]),
              diffusion_matrix=diffusion_matrix,
              target=data_loader.train_y.reshape([-1, 1, data_loader.station_number, 1]),
              batch_size=batch_size,
              sequence_length=data_loader.train_sequence_len)

# Predict
prediction = DCRNN_Obj.predict(inputs=data_loader.train_closeness.transpose([0, 2, 1, 3]),
                               diffusion_matrix=diffusion_matrix,
                               target=data_loader.train_y,
                               sequence_length=data_loader.train_sequence_len,
                               output_names=['prediction'])

# Evaluate
print('Test result', metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                                 target=data_loader.normalizer.min_max_denormal(data_loader.test_y), threshold=0))