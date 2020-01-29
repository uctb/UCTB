import os
from tqdm import tqdm
# dataset = [['Bike','NYC'],['DiDi','Xian'],['Metro','Chongqing']]
dataset = [['Metro','Chongqing']]

with open("ARIMAresult2.txt","w") as fp:

    for index in tqdm(range(len(dataset))):

        fp.write("*********************************************************\n")
        fp.write("Processing city----------------{}---using ARIMA---------".format(dataset[index]))

        f_tmp = os.popen("python -W ignore ARIMA.py --dataset {} --city {} --MergeIndex 3".format(dataset[index][0],dataset[index][1]), "r")
        # to record ouput
        fp.write(f_tmp.read()) 
        fp.flush()
        f_tmp.close()

        f_tmp = os.popen("python -W ignore ARIMA.py --dataset {} --city {} --MergeIndex 6".format(dataset[index][0],dataset[index][1]), "r")
        # to record ouput
        fp.write(f_tmp.read()) 
        fp.flush()
        f_tmp.close()

    fp.write("\n")
