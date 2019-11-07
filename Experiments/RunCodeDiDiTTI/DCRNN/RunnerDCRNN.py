import os
from tqdm import tqdm

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

    os.system(
        'python DCRNN.py --Dataset {} --Graph Correlation-Road_Distance --Group {}'.format(dataset, city))


    print("************************************************")
