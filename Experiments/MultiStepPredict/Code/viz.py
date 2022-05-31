import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def readMetric(metric_name, eva_dir, horizon_num):
    """
    Read evaluation matrices from multiple files, each row corresponds to a time step, and each column corresponds to an evaluation matrix (MAE\RMSE\MAPE) result
    output: shape is (eva_num, model_num, horizon_num)
    """
    eva_num = len(metric_name)
    model_num = len(eva_dir)
    result = np.zeros((eva_num, horizon_num, model_num))
    for m in range(model_num):
        cur_eva = pd.read_csv(eva_dir[m], header=0, index_col=0).values
        for e in range(eva_num):
            for h in range(horizon_num):
                result[e, h, m] = cur_eva[h, e]
    return result


def plot(eva_metric, metric_name, model_name, dataset_name, horizon_num):
    """
    plot
    """
    eva_num = len(metric_name)
    model_num = len(model_name)
    x = list(range(1, horizon_num+1))
    color_list = ['cornflowerblue', 'mediumorchid', 'forestgreen', 'cyan', 'darkorange', 'chocolate', 'red']
    marker_list = ['^', 'o', 'v', '<', '>', '*', 's']

    # plot
    for e in range(eva_num):
        fig = plt.figure(0, dpi=300, figsize=(8, 5))
        plt.title("multi step "+metric_name[e])
        plt.xlabel('Horizon')
        plt.ylabel(metric_name[e] + ' on '+ dataset_name)
        for m in range(model_num):
            plt.plot(x, eva_metric[e, :, m], marker=marker_list[-m], color=color_list[-m], markersize=8)
        plt.legend(model_name)
        plt.savefig('../Figure/'+ dataset_name + '_' + metric_name[e]+'.png')
        plt.close(0)


if __name__=="__main__":
    # params
    horizon_num = 12
    dataset_name = "Bike_NYC"
    metric_name = ["MAE", "RMSE", "MAPE"]
    # model_name = ["DirRec_ARIMA","DirRec_XGBoost","DirRec_DCRNN","DirRec_STMeta"]
    model_name = ["DirRec_ARIMA","DirRec_XGBoost","DirRec_DCRNN_mini","DirRec_STMeta_mini"]
    eva_dir = list(map(lambda x: "../Outputs/" + x + "-" + dataset_name + "-evaluation.csv", model_name))
    # read and plot
    eva_metric = readMetric(metric_name, eva_dir, horizon_num)
    plot(eva_metric, metric_name, model_name, dataset_name, horizon_num)
