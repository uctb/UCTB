import os
from tqdm import tqdm
# dataset = [['Bike','NYC','all','365'],['DiDi','Xian','all','all'],['Metro','Chongqing','all','all'],['ChargeStation','Beijing','all','all']]
dataset = [['Metro','Chongqing',"all",'all']]

with open("ARIMAresult16.txt","w") as fp:

    for index in tqdm(range(len(dataset))):

        # fp.write("*********************************************************\n")
        # fp.write("Processing city----------------{}---using ARIMA-------MergeIndex 3--".format(dataset[index]))
        # f_tmp = os.popen("python -W ignore ARIMA.py --dataset {} --city {} --MergeIndex 3".format(dataset[index][0],dataset[index][1]), "r")
        # # to record ouput
        # fp.write(f_tmp.read()) 
        # fp.flush()
        # f_tmp.close()

        fp.write("*********************************************************\n")
        fp.write("Processing city----------------{}---using ARIMA-------MergeIndex 3--".format(dataset[index]))
        f_tmp = os.popen("python -W ignore ARIMA.py --dataset {} --city {} --MergeIndex 3 --DataRange {} --TrainDays {}".format(dataset[index][0],dataset[index][1],dataset[index][2],dataset[index][3]), "r")
        # to record ouput
        fp.write(f_tmp.read()) 
        fp.flush()
        f_tmp.close()

    fp.write("\n")
