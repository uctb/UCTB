import json
import pickle
import numpy as np

# 支持手动输入文件名
# file_name = input("请输入文件名(需为pkl文件)，不带后缀：")
# time_slot = input("请输入时间前缀，如0429：")
file_name = 'Bike_Chicago_pred'
time_slot = '0512'
with open(file_name+'.pkl', 'rb') as f:
    data1 = pickle.load(f)

# 交换XM数据集的经纬度
station_num = len(data1['Node']['StationInfo'])
for i in range(station_num):
    info = data1['Node']['StationInfo'][i]
    latitude = info[3]
    longtitude = info[2]
    info[2] = latitude
    info[3] = longtitude

# 因为每个数据集都有自己的真实值集合，所以这部分是冗余的，可以不存。
data1["Node"]["TrafficNode"] = []
data1["Node"]["TrafficMonthlyInteraction"] = []
data1["Node"]["POI"] = []
data1["Grid"]["TrafficGrid"] = []
data1["ExternalFeature"] = []

for key in data1:
    if key not in ['Node', 'Grid', 'TimeRange', 'TimeFitness', 'Pred']:
        data1[key] = []
        
# 每个数据集内对应的内容需要保留，numpy.array需转换为list
for key1 in data1["Pred"]:
    data1["Pred"][key1]["GroundTruth"] = data1["Pred"][key1]["GroundTruth"].tolist()
    for key2 in data1["Pred"][key1]:
        if type(data1["Pred"][key1][key2]) == dict:
            data1["Pred"][key1][key2]["TrafficNode"] = data1["Pred"][key1][key2]["TrafficNode"].tolist()
            data1["Pred"][key1][key2]["traffic_data_index"] = data1["Pred"][key1][key2]["traffic_data_index"].tolist()
            if "rmse" in data1["Pred"][key1][key2]:
                data1["Pred"][key1][key2]["rmse"] = str(data1["Pred"][key1][key2]["rmse"])
            if "mape" in data1["Pred"][key1][key2]:
                data1["Pred"][key1][key2]["mape"] = str(data1["Pred"][key1][key2]["mape"])
            if "mae" in data1["Pred"][key1][key2]:
                data1["Pred"][key1][key2]["mae"] = str(data1["Pred"][key1][key2]["mae"])


jsondata = json.dumps(data1, ensure_ascii=False)
f3 = open(time_slot+'_'+file_name+'.json', 'w', encoding="utf-8")
f3.write(jsondata)

# 关闭文件，清空内存缓存区
print("已成功转换，转换后的文件为："+time_slot+'_'+file_name+'.json')
f3.close()
f.close()


