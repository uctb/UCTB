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

    print("dataset",dataset)

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation,Dataset:"{}",group:{}_DiDiTTI,mark:V1_graph_corrlation'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Road_Distance,Dataset:"{}",group:{}_DiDiTTI,mark:V1_graph_Road_Distance'.format(dataset, city))


    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation-Road_Distance,Dataset:"{}",group:{}_DiDiTTI,mark:V1_graph_graph_corrlation_Road_Distance'.format(dataset, city))


    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation,Dataset:"{}",group:{}_DiDiTTI,mark:V2_graph_corrlation'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Road_Distance,Dataset:"{}",group:{}_DiDiTTI,mark:V2_graph_Road_Distance'.format(dataset, city))


    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation-Road_Distance,Dataset:"{}",group:{}_DiDiTTI,mark:V2_graph_graph_corrlation_Road_Distance'.format(dataset, city))



    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v3.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation,Dataset:"{}",group:{}_DiDiTTI,mark:V3_graph_corrlation'.format(dataset, city))

    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v3.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Road_Distance,Dataset:"{}",group:{}_DiDiTTI,mark:V3_graph_Road_Distance'.format(dataset, city))


    os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v3.model.yml -d DiDi_RoadTTI.yml' +
              ' -p graph:Correlation-Road_Distance,Dataset:"{}",group:{}_DiDiTTI,mark:V3_graph_graph_corrlation_Road_Distance'.format(dataset, city))
