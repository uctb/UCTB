import os
import numpy as np

from local_path import *
from EvalClass.Accuracy import Accuracy

prediction_file = 'MGCN_{}_{}_K1L1_V0.npy'
target_file = '{}_target.npy'

city = ['NYC', 'Chicago', 'DC']
graph = ['D', 'C', 'I']

for c in city:
    prediction = []
    for g in graph:
        prediction.append(np.load(os.path.join(data_dir, prediction_file.format(c, g))))

    naive_average_prediction = np.mean(prediction, axis=0)

    target = np.load(os.path.join(data_dir, target_file.format(c)))

    print(c, 'RMSE naive agerage', Accuracy.RMSE(p=naive_average_prediction, t=target, threshold=0))

    weighted_average_rmse = []

    for i in [e/10.0 for e in range(11)]:
        for j in [e/10.0 for e in range(11)]:
            for k in [e/10.0 for e in range(11)]:
                if i + j + k == 1.0:

                    weighted_average_prediction = i * prediction[0] + j * prediction[1] + k * prediction[2]

                    test_rmse = Accuracy.RMSE(p=weighted_average_prediction, t=target, threshold=0)

                    weighted_average_rmse.append(test_rmse)

                    # print(c, i, j, k, test_rmse)

    print(c, 'Best Weighted Average', min(weighted_average_rmse))