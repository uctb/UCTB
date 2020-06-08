import os
import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import DCRNN
from UCTB.evaluation import metric

from UCTB.preprocess.GraphGenerator import GraphGenerator

class my_data_loader(NodeTrafficLoader):

    def __init__(self, **inner_args):

        # graphBuilder = GraphGenerator(inner_args['graph'],
        #                      dataset=inner_args['dataset'],
        #                      MergeIndex=inner_args['MergeIndex'],
        #                      MergeWay=inner_args['MergeWay'],
        #                      city=inner_args['city'],
        #                      data_range=inner_args['data_range'],
        #                      train_data_length=inner_args['train_data_length'],
        #                     #  test_ratio=0.1,
        #                      threshold_distance=inner_args['threshold_distance'],
        #                      threshold_correlation=inner_args['threshold_correlation'],
        #                      threshold_interaction=inner_args['threshold_interaction'],
        #                     #  threshold_neighbour=inner_args['threshold_neighbour'],
        #                      normalize=inner_args['normalize'])
        # self.AM = graphBuilder.AM
        # self.LM = graphBuilder.LM

        super(my_data_loader, self).__init__(**inner_args) # [!INFO] Init NodeTrafficLoader
        
        # Import the Class:GraphGenerator
        # Call GraphGenerator to initialize and generate LM
        graph = inner_args['graph']
        graphBuilder = GraphGenerator(graph,
                             dataset = self.dataset,
                             train_data = self.train_data,
                             traffic_data_index = self.traffic_data_index,
                             train_test_ratio = self.train_test_ratio,
                             threshold_distance=inner_args['threshold_distance'],
                             threshold_correlation=inner_args['threshold_correlation'],
                             threshold_interaction=inner_args['threshold_interaction']
                             )
        self.AM = graphBuilder.AM
        self.LM = graphBuilder.LM

    def diffusion_matrix(self, filter_type='random_walk'):
        # print("Threshold for inter: ",threshold_interaction)
        # print ("daily_slots: ", self.daily_slots)
        def calculate_random_walk_matrix(adjacent_mx):
            d = np.array(adjacent_mx.sum(1))
            d_inv = np.power(d, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = np.diag(d_inv)
            random_walk_mx = d_mat_inv.dot(adjacent_mx)
            return random_walk_mx
        # assert len(self.AM) == 1



        diffusion_matrix = []
        if filter_type == "random_walk":
            diffusion_matrix.append(calculate_random_walk_matrix(self.AM[0]).T)
        elif filter_type == "dual_random_walk":
            diffusion_matrix.append(calculate_random_walk_matrix(self.AM[0]).T)
            diffusion_matrix.append(calculate_random_walk_matrix(self.AM[0].T).T)
        return np.array(diffusion_matrix, dtype=np.float32)


def param_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='DiDi')
    parser.add_argument('--City', default='Chengdu')
    # network parameter
    parser.add_argument('--CT', default='6', type=int)
    parser.add_argument('--PT', default='7', type=int)
    parser.add_argument('--TT', default='4', type=int)
    parser.add_argument('--K', default='1', type=int)
    parser.add_argument('--L', default='1', type=int)
    parser.add_argument('--Graph', default='Distance-Correlation-Interaction')
    parser.add_argument('--LSTMUnits', default='64', type=int)
    parser.add_argument('--LSTMLayers', default='3', type=int)
    # Training data parameters
    parser.add_argument('--DataRange', default='All')
    parser.add_argument('--TrainDays', default='365')
    # Graph parameter
    parser.add_argument('--TC', default='0', type=float)
    parser.add_argument('--TD', default='1000', type=float)
    parser.add_argument('--TI', default='500', type=float)
    # training parameters
    parser.add_argument('--Epoch', default='5000', type=int)
    parser.add_argument('--Train', default='True')
    parser.add_argument('--lr', default='5e-4', type=float)
    parser.add_argument('--ESlength', default='50', type=int)
    parser.add_argument('--patience', default='0.1', type=float)
    parser.add_argument('--BatchSize', default='64', type=int)
    # device parameter
    parser.add_argument('--Device', default='0', type=str)
    # version control
    parser.add_argument('--Group', default='DebugGroup')
    parser.add_argument('--CodeVersion', default='ST_MGCN_Debug')
    # Merge times
    parser.add_argument('--MergeIndex', default=6, type=int)
    return parser


parser = param_parser()
args = parser.parse_args()

model_dir = os.path.join('model_dir', args.City)
code_version = 'DCRNN_{}_K{}L{}_{}_F{}'.format(''.join([e[0] for e in args.Graph.split('-')]),
                                           args.K, args.L, args.CodeVersion, int(args.MergeIndex)*5)

data_loader = my_data_loader(dataset=args.Dataset, city=args.City,
                             data_range=args.DataRange, train_data_length=args.TrainDays,
                             closeness_len=int(args.CT), period_len=int(args.PT), trend_len=int(args.TT),
                             threshold_interaction=args.TI, threshold_distance=args.TD,
                             threshold_correlation=args.TC, graph=args.Graph, with_lm=True, normalize=True, MergeIndex=args.MergeIndex,
                             MergeWay="max" if args.Dataset == "ChargeStation" else "sum")

print('Code version', args.Dataset, args.City, code_version)

print('Number of training samples', data_loader.train_sequence_len)

diffusion_matrix = data_loader.diffusion_matrix()

DCRNN_Obj = DCRNN(num_nodes=data_loader.station_number,
                  num_diffusion_matrix=diffusion_matrix.shape[0],
                  num_rnn_units=args.LSTMUnits,
                  num_rnn_layers=args.LSTMLayers,
                  max_diffusion_step=args.K,
                  seq_len=data_loader.closeness_len + data_loader.period_len + data_loader.trend_len,
                  use_curriculum_learning=False,
                  input_dim=1,
                  output_dim=1,
                  cl_decay_steps=1000,
                  target_len=1,
                  lr=args.lr,
                  epsilon=1e-3,
                  optimizer_name='Adam',
                  code_version=code_version,
                  model_dir=model_dir,
                  gpu_device=args.Device)

# Build tf-graph
DCRNN_Obj.build()

print('Number of trainable parameters', DCRNN_Obj.trainable_vars)

# Training
DCRNN_Obj.fit(inputs=
                # np.concatenate((
                    # data_loader.train_trend.transpose([0, 2, 1, 3]),
                    # data_loader.train_period.transpose([0, 2, 1, 3]),
                    data_loader.train_closeness.transpose([0, 2, 1, 3]),
                # ), axis=1),
              diffusion_matrix=diffusion_matrix,
              target=data_loader.train_y.reshape([-1, 1, data_loader.station_number, 1]),
              batch_size=args.BatchSize,
              sequence_length=data_loader.train_sequence_len,
              early_stop_length=args.ESlength,
              max_epoch=args.Epoch)

# Predict
prediction = DCRNN_Obj.predict(inputs=
                    # np.concatenate((
                    # data_loader.test_trend.transpose([0, 2, 1, 3]),
                    # data_loader.test_period.transpose([0, 2, 1, 3]),
                    data_loader.test_closeness.transpose([0, 2, 1, 3]),
                    # ), axis=1),
                               diffusion_matrix=diffusion_matrix,
                               target=data_loader.test_y.reshape([-1, 1, data_loader.station_number, 1]),
                               sequence_length=data_loader.test_sequence_len,
                               output_names=['prediction'])

# Evaluate
print('Test result', metric.rmse(prediction=data_loader.normalizer.min_max_denormal(prediction['prediction']),
                                 target=data_loader.normalizer.min_max_denormal(data_loader.test_y.transpose([0, 2, 1])),
                                 threshold=0))

val_loss = DCRNN_Obj.load_event_scalar('val_loss')

best_val_loss = min([e[-1] for e in val_loss])

best_val_loss = data_loader.normalizer.min_max_denormal(best_val_loss)

print('Best val result', best_val_loss)

time_consumption = [val_loss[e][0] - val_loss[e-1][0] for e in range(1, len(val_loss))]
time_consumption = sum([e for e in time_consumption if e < (min(time_consumption) * 10)]) / 3600
print('Converged using %.2f hour / %s epochs' % (time_consumption, DCRNN_Obj._global_step))