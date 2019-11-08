import os
from tqdm import tqdm

#############################################
# BenchMark DiDi TTI Dataset
#############################################


# all  *.pkl data should be in data_root
# Chai :  D:\\LiyueChen\\Data\\
# Leye1 : E:\\chenliyue\Data\\
# lychen : E:\\滴滴数据集\\已处理-滴滴-城市交通指数-6个城市-一年数据
data_root = r'../../Data'


# there are five cities in this experiment
city_ls = ['Chengdu', 'Haikou', 'Jinan', 'Shenzhen', 'Suzhou', 'Xian']


for index, city in enumerate(tqdm(city_ls)):

    dataset = os.path.join(data_root, "DiDi_{}_RoadTTI.pkl".format(city))

    print("dataset", dataset)

    # run AMulti_GCLSTM V1 V2 V3
    #gcn_k:1,gclstm_layers:1,batch_size:16
    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation,Dataset:"{}",group:{}_AMulti,mark:V1'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Road_Distance,Dataset:"{}",group:{}_AMulti,mark:V1'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation-Road_Distance,Dataset:"{}",group:{}_AMulti,mark:V1'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation,Dataset:"{}",group:{}_AMulti,mark:V2'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Road_Distance,Dataset:"{}",group:{}_AMulti,mark:V2'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation-Road_Distance,Dataset:"{}",group:{}_AMulti,mark:V2'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v3.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation,Dataset:"{}",group:{}_AMulti,mark:V3'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v3.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Road_Distance,Dataset:"{}",group:{}_AMulti,mark:V3'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v3.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation-Road_Distance,Dataset:"{}",group:{}_AMulti,mark:V3'.format(dataset, city))

    # run TMeta-LSTM-GAL
    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d DiDi_RoadTTI.yml' +
              ' -p st_method:LSTM,graph:Road_Distance,Dataset:"{}",group:{}_TMeta,mark:V1'.format(dataset, city))
