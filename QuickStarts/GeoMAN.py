from UCTB.dataset import NodeTrafficLoader
from UCTB.preprocess import MoveSample
from UCTB.model import GeoMAN
from UCTB.evaluation import metric
import numpy as np


class GeoMAN_DataLoader(NodeTrafficLoader):
    def __init__(self, node_id=0, input_steps=12, output_steps=1, **kwargs):
        super(GeoMAN_DataLoader, self).__init__(closeness_len=input_steps,
                                                period_len=input_steps,
                                                trend_len=input_steps,  # TODO modify this
                                                target_length=output_steps,
                                                **kwargs)
        self.node_id = node_id  # specify a node to train and predict
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.move_ef = MoveSample(input_steps, 0, 0, output_steps)

        self.train_local_features, self.train_global_features, self.train_local_attn_states, self.train_global_attn_states, self.train_external_features, self.train_y = self.process_data(
            self.train_trend, self.train_period, self.train_closeness, self.train_ef, self.train_y)

        self.test_local_features, self.test_global_features, self.test_local_attn_states, self.test_global_attn_states, self.test_external_features, self.test_y = self.process_data(
            self.test_trend, self.test_period, self.test_closeness, self.test_ef, self.test_y)
        self.input_dim = self.train_local_features.shape[2]
        self.output_dim = self.train_y.shape[-1]
        self.train_seq_len = self.train_external_features.shape[0]  # TODO max? or min? update new sequence length
        self.test_seq_len = self.test_external_features.shape[0]

    def process_data(self, trend, period, closeness, ef, y):
        # apply timestep to external features, which will make its length shorter
        _, ext_features = self.move_ef.general_move_sample(ef)
        seq_len = ext_features.shape[0]

        global_attn_states = [d[:seq_len] for d in
                              [closeness, period]]  # , trend]]  # TODO add trend # clip length to align

        global_features = global_attn_states[0]  # target to predict is closeness
        global_features = global_features.transpose([0, 2, 1, 3]).reshape(-1, self.input_steps, self.station_number)  # (batch_size, n_steps_encoder, n_sensors)

        local_features = [f[:, self.node_id, :, :] for f in global_attn_states]  # pick the records of that node
        local_features = np.concatenate(local_features, axis=2)  # (batch_size, n_steps_encoder, n_input_encoder)

        local_attn_states = local_features.transpose([0, 2, 1])  # (batch_size, n_input_encoder, n_steps_encoder)

        global_attn_states = np.concatenate(global_attn_states, axis=3).transpose(
            [0, 1, 3, 2])  # (batch_size, n_sensors, n_input_encoder, n_steps_encoder)

        y = y[:seq_len, self.node_id, :]  # pick the records of that node
        y = y.reshape(-1, self.output_steps, y.shape[-1])
        return local_features, global_features, local_attn_states, global_attn_states, ext_features, y



import time
start_time = time.time()

data_loader = GeoMAN_DataLoader(dataset='Bike', city='NYC', node_id=0, input_steps=12, output_steps=1)
model = GeoMAN(total_sensers=data_loader.station_number,
               input_dim=data_loader.input_dim,
               external_dim=data_loader.external_dim,
               output_dim=data_loader.output_dim,
               input_steps=data_loader.input_steps,
               output_steps=data_loader.output_steps)

model.build()
model.fit(local_features=data_loader.train_local_features,
          global_features=data_loader.train_global_features,
          local_attn_states=data_loader.train_local_attn_states,
          global_attn_states=data_loader.train_global_attn_states,
          external_features=data_loader.train_external_features,
          targets=data_loader.train_y,
          sequence_length=data_loader.train_seq_len)

results = model.predict(local_features=data_loader.test_local_features,
              global_features=data_loader.test_global_features,
              local_attn_states=data_loader.test_local_attn_states,
              global_attn_states=data_loader.test_global_attn_states,
              external_features=data_loader.test_external_features,
              targets=data_loader.test_y,
              sequence_length=data_loader.test_seq_len)

print('RMSE', metric.rmse(results['prediction'], data_loader.test_y, threshold=0))
print("--- %s seconds ---" % (time.time() - start_time))