import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

shared_params_st_mgcn = ('python ST_MGCN_Obj.py '
                         '--Dataset Metro '
                         '--CT 6 '
                         '--PT 7 '
                         '--TT 4 '
                         '--LSTMUnits 64 '
                         '--LSTMLayers 3 '
                         '--DataRange All '
                         '--TrainDays 365 '
                         '--TC 0.7 '
                         '--TD 5000 '
                         '--TI 30 '
                         '--Epoch 10000 '
                         '--Train True '
                         '--lr 1e-4 '
                         '--patience 0.1 '
                         '--ESlength 100 '
                         '--BatchSize 16 '
                         '--Device 1 ')

#############################################
# BenchMark DiDi TTI Dataset
#############################################


# all  *.pkl data should be in data_root
# Chai :  D:\\LiyueChen\\Data\\
# Leye1 : E:\\chenliyue\Data\\
# lychen : E:\\滴滴数据集\\已处理-滴滴-城市交通指数-6个城市-一年数据
data_root = r'E:\\chenliyue\Data\\'


# there are five cities in this experiment
city_ls = ['Chengdu', 'Haikou', 'Jinan', 'Shenzhen', 'Suzhou', 'Xian']


for index, city in enumerate(tqdm(city_ls)):

    dataset = os.path.join(data_root, "DiDi_{}_RoadTTI.pkl".format(city))
    print("************************************************")
    print("dataset", dataset)

    os.system('python ST_MGCN_Obj.py --Dataset {} --Graph Correlation-Road_Distance --Group {}'.format(dataset,city))

    print("************************************************")
