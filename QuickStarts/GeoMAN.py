from UCTB.dataset import NodeTrafficLoader
from UCTB.preprocess import MoveSample
from UCTB.model import GeoMAN
from UCTB.evaluation import metric
import numpy as np
import time


class GeoMAN_DataLoader(NodeTrafficLoader):
    def __init__(self, input_steps=12, output_steps=1, **kwargs):
        """A wrapper of ``NodeTrafficLoader`` to make its data form compatible with GeoMAN's inputs.

        Args:
            input_steps (int): The length of historical input data, a.k.a, input timesteps. Default: 12
            output_steps (int): The number of steps that need prediction by one piece of history data, a.k.a,
                output timesteps. Have to be 1 now. Default: 1
            **kwargs (dict): Used to pass other parameters to class ``NodeTrafficLoader``.

        Attributes:
            train_local_features (list): A list, where each element corresponds to the ``local_features`` in GeoMAN's
                feed dict of one sensor (node)  and the length of list is ``station_number``. Uses indexes of it to
                specify a target sensor, e.g., ``train_local_features[i]`` for sensor ``i``.
            train_local_attn_states (list): A list containing each sensor's ``local_attn_states``
            train_y (list): A list containing each sensor's label ndarray.
            train_seq_len (int): The total sample number of training data set, which will be used in mini-batch training.
        """
        super(GeoMAN_DataLoader, self).__init__(closeness_len=input_steps,
                                                period_len=input_steps,
                                                trend_len=input_steps,
                                                target_length=output_steps,
                                                **kwargs)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.move_ef = MoveSample(input_steps, 0, 0, output_steps)

        self.train_local_features, self.train_global_features, self.train_local_attn_states, self.train_global_attn_states, self.train_external_features, self.train_y = self.process_data(
            self.train_trend, self.train_period, self.train_closeness, self.train_ef, self.train_y)

        self.test_local_features, self.test_global_features, self.test_local_attn_states, self.test_global_attn_states, self.test_external_features, self.test_y = self.process_data(
            self.test_trend, self.test_period, self.test_closeness, self.test_ef, self.test_y)
        self.input_dim = self.train_local_features[0].shape[-1]
        self.output_dim = self.train_y[0].shape[-1]
        self.train_seq_len = self.train_external_features.shape[0]
        self.test_seq_len = self.test_external_features.shape[0]

    def process_data(self, trend, period, closeness, ef, y):
        """Process features to GeoMAN's acceptable forms

        Different from other models, GeoMAN needs all the inputs that have the same timesteps, so we generate
        ``closeness``, ``period`` and ``trend`` from a fixed length. After that, we simply concatenate these three
        features into a single matrix ``global_attn_states``. Based on it, we can eventually construct all the inputs
        of GeoMAN, including ``local_features``, ``global_features``, ``local_attn_states`` and
        ``global_attn_states``. Moreover, since the original eternal features in ``NodeTrafficLoader`` are timeless,
        we handle them with ``move_ef`` to generate timesteps as a workaround.

        """
        # apply timestep to external features, which will make its length shorter
        _, ext_features = self.move_ef.general_move_sample(ef)
        seq_len = ext_features.shape[0]

        global_attn_states = [d[:seq_len] for d in [closeness, period, trend]]  # clip length to align

        global_features = global_attn_states[0]  # target to predict is closeness
        global_features = global_features.transpose([0, 2, 1, 3])
        global_features = global_features.reshape(-1, self.input_steps,
                                                  self.station_number)  # (batch_size, n_steps_encoder, n_sensors)
        global_attn_states = np.concatenate(global_attn_states, axis=3)
        local_features = np.split(global_attn_states, self.station_number,
                                  axis=1)  # [(batch_size, n_steps_encoder, ), ...] list of nodes
        local_features = [d.squeeze(1) for d in local_features]
        global_attn_states = global_attn_states.transpose(
            [0, 1, 3, 2])  # (batch_size, n_sensors, n_input_encoder, n_steps_encoder)
        local_attn_states = np.split(global_attn_states, self.station_number,
                                     axis=1)  # [(batch_size, n_input_encoder, n_steps_encoder)
        local_attn_states = [d.squeeze(1) for d in local_attn_states]

        y = np.split(y[:seq_len], self.station_number, axis=1)
        return local_features, global_features, local_attn_states, global_attn_states, ext_features, y


data_loader = GeoMAN_DataLoader(dataset='Bike', city='NYC', input_steps=12, output_steps=1)
model = GeoMAN(total_sensers=data_loader.station_number,
               input_dim=data_loader.input_dim,
               external_dim=data_loader.external_dim,
               output_dim=data_loader.output_dim,
               input_steps=data_loader.input_steps,
               output_steps=data_loader.output_steps)
model.build()
# training and evaluation
results = []
for node in range(data_loader.station_number):
    each_time = time.time()
    model._code_version = str(node)  # to train different model for different node
    model.fit(local_features=data_loader.train_local_features[node],
              global_features=data_loader.train_global_features,
              local_attn_states=data_loader.train_local_attn_states[node],
              global_attn_states=data_loader.train_global_attn_states,
              external_features=data_loader.train_external_features,
              targets=data_loader.train_y[node],
              sequence_length=data_loader.train_seq_len)

    pred = model.predict(local_features=data_loader.test_local_features[node],
                         global_features=data_loader.test_global_features,
                         local_attn_states=data_loader.test_local_attn_states[node],
                         global_attn_states=data_loader.test_global_attn_states,
                         external_features=data_loader.test_external_features,
                         targets=data_loader.test_y[node],
                         sequence_length=data_loader.test_seq_len)
    results.append(metric.rmse(pred['prediction'], data_loader.test_y[node]))
    seconds = int(time.time() - each_time)
    print('[Node {}] - {}s - RMSE: {}'.format(node, seconds, results[-1]))

    # randomize weights again for next node
    model._session.run(model._variable_init)

print('Overall average RMSE: {}'.format(np.mean(results)))
