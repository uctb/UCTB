import copy
import pickle
import os
import numpy as np

def make_predict_dataset(data_loader, predict_val, method):
    if "Pred" not in data_loader.dataset.data.keys():
        data_loader.dataset.data['Pred'] = {}
    data_loader.dataset.data['Pred'][method] = {}
    data_loader.dataset.data['Pred'][method]["traffic_data_index"] = data_loader.traffic_data_index
    data_loader.dataset.data['Pred'][method]["TrafficNode"] = np.squeeze(
        predict_val)

    with open(os.path.join(os.path.abspath(os.getcwd()), "{}_pred.pkl".format(data_loader.dataset.city)), "wb") as fp:
        pickle.dump(data_loader.dataset.data, fp)
