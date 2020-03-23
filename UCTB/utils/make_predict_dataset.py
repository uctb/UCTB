import copy
import pickle
import os
import numpy as np


def save_predict_in_dataset(data_loader, predict_val, method):
    data_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data')

    original_file = os.path.join(data_dir, "{}_{}.pkl".format(
        data_loader.dataset.dataset, data_loader.dataset.city))
    file_name = os.path.join(data_dir, "{}_{}_pred.pkl".format(
        data_loader.dataset.dataset, data_loader.dataset.city))

    if os.path.exists(file_name):
        with open(file_name, "rb") as fp:
            pred_data = pickle.load(fp)
    else:
        with open(original_file, "rb") as fp:
            pred_data = pickle.load(fp)
            pred_data['Pred'] = {}

    loader_id = data_loader.loader_id
    if loader_id not in pred_data["Pred"].keys():
        pred_data['Pred'][loader_id] = {}

    pred_data['Pred'][loader_id]["GroundTruth"] = np.squeeze(
        data_loader.test_y)

    if loader_id.endswith("N"):
        # use NodeTrafficLoader
        pred_data['Pred'][loader_id][method] = {}
        pred_data['Pred'][loader_id][method]["traffic_data_index"] = data_loader.traffic_data_index
        pred_data['Pred'][loader_id][method]["TrafficNode"] = np.squeeze(
            predict_val)

    if loader_id.endswith("G"):
        # use GridTrafficLoader
        pred_data['Pred'][method]["TrafficGrid"] = np.squeeze(predict_val)

    with open(file_name, "wb") as fp:
        pickle.dump(pred_data, fp)
