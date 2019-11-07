import os
from tqdm import tqdm

# ###############################################
# # BenchMark DiDi TTI data
# ###############################################
#




# all  *.pkl data should be in data_root
# Chai :  D:\\LiyueChen\\Data\\
# Leye1 : E:\\chenliyue\Data\\
# lychen : E:\\滴滴数据集\\已处理-滴滴-城市交通指数-6个城市-一年数据
data_root = 'E:\\滴滴数据集\\已处理-滴滴-城市交通指数-6个城市-一年数据'


#there are five cities in this experiment
city_ls = ['Chengdu', 'Haikou', 'Jinan', 'Shenzhen', 'Suzhou', 'Xian']


with open("ARIMAresult.txt","w") as fp:

    for index in tqdm(range(len(city_ls))):

        fp.write("*********************************************************\n")
        fp.write("Processing city----------------{}---using ARIMA---------".format(city_ls[index]))

        currentPath = os.path.join(data_root,"DiDi_{}_RoadTTI.pkl".format(city_ls[index]))

        f_tmp = os.popen("python -W ignore ARIMA.py --Dataset {}".format(currentPath), "r")
        # to record ouput
        fp.write(f_tmp.read()) 
        fp.flush()
        f_tmp.close()
        fp.write("*********************************************************\n")

    fp.write("\n")
