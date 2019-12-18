import os
from tqdm import tqdm

#############################################
# BenchMark DiDi TTI Dataset
#############################################

# all  *.pkl data should be in data_root
data_root = r'../ReciData'


# there are five cities in this experiment
#city_ls = ['Chengdu', 'Haikou', 'Jinan', 'Shenzhen', 'Suzhou', 'Xian']
city_ls = ['Haikou']
thresholdMap = {'Chengdu': 5000,
                'Haikou': 3500,
                'Jinan': 5000,
                'Shenzhen': 6500,
                'Suzhou': 10500,
                'Xian': 6000}


for index, city in enumerate(tqdm(city_ls)):

    dataset = os.path.join(data_root, "DiDi_{}_RoadTTI.pkl".format(city))

    print("dataset", dataset)

    # # run lrDecay
    # naive method
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDi_RoadTTI.yml'
              ' -p batch_size:64,graph:Road_Distance,st_method:LSTM,threshold_road_distance:{},Dataset:"{}",group:{}_lrDecay,model_version:naive,mark:V0'.format(thresholdMap[city], dataset, city))

    # # naive method lr:1e-4
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDi_RoadTTI.yml'
              ' -p lr:1e-4,batch_size:64,graph:Road_Distance,st_method:LSTM,threshold_road_distance:{},Dataset:"{}",group:{}_lrDecay,model_version:naive_lr1e-4,mark:V0'.format(thresholdMap[city], dataset, city))
    
    # # naive method lr:5e-4
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDi_RoadTTI.yml'
              ' -p lr:5e-4,batch_size:64,graph:Road_Distance,st_method:LSTM,threshold_road_distance:{},Dataset:"{}",group:{}_lrDecay,model_version:naive_lr5e-4,mark:V0'.format(thresholdMap[city], dataset, city))

    # exp method
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDi_RoadTTI.yml -l exp_decay_param.yml'
              ' -p early_stop_patience:1000,batch_size:64,graph:Road_Distance,st_method:LSTM,threshold_road_distance:{},Dataset:"{}",group:{}_lrDecay,model_version:exponential,mark:V0'.format(thresholdMap[city], dataset, city))

    # cos method
    os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDi_RoadTTI.yml -l cosine_decay_param.yml'
              ' -p early_stop_patience:1000,batch_size:64,graph:Road_Distance,st_method:LSTM,threshold_road_distance:{},Dataset:"{}",group:{}_lrDecay,model_version:cosine,mark:V0'.format(thresholdMap[city], dataset, city))
