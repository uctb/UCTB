import copy
import pickle as pkl
import os
import numpy as np
from dateutil.parser import parse
from datetime import timedelta
import pandas as pd

def save_predict_in_dataset(data_loader, predict_val, method):
    data_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data')

    original_file = os.path.join(data_dir, "{}_{}.pkl".format(
        data_loader.dataset.dataset, data_loader.dataset.city))
    file_name = os.path.join(data_dir, "{}_{}_pred.pkl".format(
        data_loader.dataset.dataset, data_loader.dataset.city))

    if os.path.exists(file_name):
        with open(file_name, "rb") as fp:
            pred_data = pkl.load(fp)
    else:
        with open(original_file, "rb") as fp:
            pred_data = pkl.load(fp)
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
        pred_data['Pred'][loader_id][method]["TrafficNode"] = np.squeeze(predict_val)

    if loader_id.endswith("G"):
        # use GridTrafficLoader
        pred_data['Pred'][loader_id][method] = {}
        pred_data['Pred'][loader_id][method]["TrafficGrid"] = np.squeeze(predict_val)

    with open(file_name, "wb") as fp:
        pkl.dump(pred_data, fp)
    
def save_predict_and_graph_in_tsv_and_array(data_loader, prediction,args_list, is_graph=False, output_dir='output',graph=None):

    #TODO: add an argument used to choose whether train set or test set is saved or both.
    #TODO: text form to save information of time_fitness and time_range

    # get access to original dataset
    
    dataset = data_loader.dataset

    end_date = dataset.time_range[1]
    loader_id = data_loader.loader_id
    
    # parse parameters setting through loader_id
    
    data_range, train_data_length, test_ratio, closeness_len, period_len, trend_len, time_fitness,_= loader_id.split('_')
    
    # get reasonable range of data we use
    
    if type(data_range) is str and data_range.lower().startswith("0."):
        data_range = float(data_range)
    if type(data_range) is str and data_range.lower() == 'all':
        data_range = [0, len(data_loader.dataset.node_traffic)]
    elif type(data_range) is float:
        data_range = [0, int(data_range * len(data_loader.dataset.node_traffic))]
    else:
        data_range = [int(data_range[0] * data_loader.daily_slots), int(data_range[1] * data_loader.daily_slots)]
    
    # get total number of time slots we use
    
    number_of_ts = data_range[1] - data_range[0]
    test_start_index = int(data_loader.train_test_ratio[0] * number_of_ts)
    assert int(number_of_ts-test_start_index) == data_loader.test_y.shape[0]
    td = timedelta(minutes=int(time_fitness))

    # obtain begining and ending date of test set

    test_set_start_date = (parse(end_date) - td*(len(dataset.node_traffic)-test_start_index)).strftime('%Y-%m-%d %H:%M:%S')
    test_set_end_date = (parse(end_date) - td*(len(dataset.node_traffic)-number_of_ts)).strftime('%Y-%m-%d %H:%M:%S')
    traffic_data_index = list(data_loader.traffic_data_index)
    station_info = []
    for ind in traffic_data_index:
        station_info.append(dataset.node_station_info[ind])


    if type(args_list) == list:
        method = ''.join(args_list)
    else:
        method = args_list

    # get dataset name
    file_name_without_extension = '{}_{}_{}'.format(dataset.dataset , dataset.city , method)

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gt_list = data_loader.test_y.reshape([-1,207])
    pd_list = prediction
    gt_df = pd.DataFrame(gt_list)
    pd_df = pd.DataFrame(pd_list)
    gt_df = gt_df.transpose()
    pd_df = pd_df.transpose()

    station_info_df = pd.DataFrame(station_info)
    station_info_df = station_info_df.drop(station_info_df.columns[[0, 1, 4]], axis=1)
    gt_path = os.path.join(output_dir,file_name_without_extension + '_gt.tsv')
    pd_path = os.path.join(output_dir,file_name_without_extension + '_pd.tsv')
    station_info_path = os.path.join(output_dir,file_name_without_extension + '_station_info.tsv')
    
    # generate npy files
    
    np.save(os.path.join(output_dir,file_name_without_extension + '_gt.npy'),gt_list)
    np.save(os.path.join(output_dir,file_name_without_extension + '_pd.npy'),pd_list)
    
    # generate graph
    
    if is_graph:
        np.save(os.path.join(output_dir,file_name_without_extension + '_graph.npy'),graph)
    
    # generate tsv files

    try:
        gt_df.to_csv(gt_path, sep='\t', index=False, header=False)
        pd_df.to_csv(pd_path, sep='\t', index=False, header=False)
        station_info_df.to_csv(station_info_path, sep='\t', index=False,
                               header=False)
        print("TSV files generated successfully!")
    except Exception as e:
        print("Error while generating TSV files:", e)

    print('start time:{};end time:{}'.format(test_set_start_date,test_set_end_date))
    print('time fitness:{}'.format(int(time_fitness)))
    return [test_set_start_date,test_set_end_date], int(time_fitness)

def save_predict_and_graph_in_tsv_and_array(data_loader, prediction,args_list, output_dir='output',graph=None):

    #TODO: add an argument used to choose whether train set or test set is saved or both.
    #TODO: text form to save information of time_fitness and time_range

    # get access to original dataset

    dataset = data_loader.dataset

    end_date = dataset.time_range[1]
    loader_id = data_loader.loader_id
    # parse parameters setting through loader_id

    data_range, train_data_length, test_ratio, closeness_len, period_len, trend_len, time_fitness,_= loader_id.split('_')
    
    # get reasonable range of data we use

    if type(data_range) is str and data_range.lower().startswith("0."):
        data_range = float(data_range)
    if type(data_range) is str and data_range.lower() == 'all':
        data_range = [0, len(data_loader.dataset.node_traffic)]
    elif type(data_range) is float:
        data_range = [0, int(data_range * len(data_loader.dataset.node_traffic))]
    else:
        data_range = [int(data_range[0] * data_loader.daily_slots), int(data_range[1] * data_loader.daily_slots)]
    
    # get total number of time slots we use

    number_of_ts = data_range[1] - data_range[0]
    
    test_start_index = data_range[0]+int(data_loader.train_test_ratio[0] * number_of_ts)
    assert int(data_range[1]-test_start_index) == data_loader.test_y.shape[0]

    # obtain begining and ending date of test set

    td = timedelta(minutes=int(time_fitness))
    test_set_start_date = (parse(end_date) - td*(len(dataset.node_traffic)-test_start_index+1)).strftime('%Y-%m-%d %H:%M:%S')
    test_set_end_date = (parse(end_date) - td*(len(dataset.node_traffic)-data_range[1]+1)).strftime('%Y-%m-%d %H:%M:%S')

    traffic_data_index = list(data_loader.traffic_data_index)
    station_info = []
    for ind in traffic_data_index:
        station_info.append(dataset.node_station_info[ind])


    if type(args_list) == list:
        method = ''.join(args_list)
    else:
        method = args_list

    # set output tsv name
    file_name_without_extension = '{}_{}_{}'.format(dataset.dataset , dataset.city , method)

    # generate pred and gt station_info in tsv files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gt_list = data_loader.test_y.reshape([-1,207])
    pd_list = prediction
    gt_df = pd.DataFrame(gt_list)
    pd_df = pd.DataFrame(pd_list)
    gt_df = gt_df.transpose()
    pd_df = pd_df.transpose()
    station_info_df = pd.DataFrame(station_info)
    station_info_df = station_info_df.drop(station_info_df.columns[[0, 1, 4]], axis=1)
    gt_path = os.path.join(output_dir,file_name_without_extension + '_gt.tsv')
    pd_path = os.path.join(output_dir,file_name_without_extension + '_pd.tsv')
    station_info_path = os.path.join(output_dir,file_name_without_extension + '_station_info.tsv')

    # generate pred and gt in npy files and station_info in pkl file

    np.save(os.path.join(output_dir,file_name_without_extension + '_gt.npy'),gt_list)
    np.save(os.path.join(output_dir,file_name_without_extension + '_pd.npy'),pd_list)
    with open(os.path.join(output_dir,file_name_without_extension + '_station_info.pkl'),'wb') as fp:
        pkl.dump(station_info,fp)
        
    # generate graph files

    if graph is None:
        np.save(os.path.join(output_dir,file_name_without_extension + '_graph.npy'),graph)

    # save tsv files

    try:
        gt_df.to_csv(gt_path, sep='\t', index=False, header=False)
        pd_df.to_csv(pd_path, sep='\t', index=False, header=False)
        station_info_df.to_csv(station_info_path, sep='\t', index=False,
                               header=False)
        print("TSV files generated successfully!")
    except Exception as e:
        print("Error while generating TSV files:", e)

    print('start time:{};end time:{}'.format(test_set_start_date,test_set_end_date))
    print('time fitness:{}'.format(int(time_fitness)))
    return [test_set_start_date,test_set_end_date], int(time_fitness)