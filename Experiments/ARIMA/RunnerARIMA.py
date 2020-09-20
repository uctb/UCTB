import os
from tqdm import tqdm
# dataset = [['Bike','NYC','all','365','sum','0.1'],['DiDi','Xian','all','all','sum','0.1'],
# ['Metro','Chongqing','all','all','sum','0.1'],['ChargeStation','Beijing','all','all','max','0.1'],
# ['METR','LA','all','all','average','0.2'],['PEMS','BAY','all','all','average','0.2']]
dataset = [['METR','LA','all','all','average','0.2'],['PEMS','BAY','all','all','average','0.2']]

with open("ARIMAresult3.txt","w") as fp:

    for index in tqdm(range(len(dataset))):

        fp.write("*********************************************************\n")
        fp.write("Processing city----------------{}---using ARIMA-------MergeIndex 12 --".format(dataset[index]))
        f_tmp = os.popen("python -W ignore ARIMA.py --dataset {} --city {} --MergeIndex 12 --DataRange {} --TrainDays {} --MergeWay {} --test_ratio {}".format(dataset[index][0],dataset[index][1],dataset[index][2],dataset[index][3],dataset[index][4],dataset[index][5]), "r")
        # to record ouput
        fp.write(f_tmp.read()) 
        fp.flush()
        f_tmp.close()

    fp.write("\n")
