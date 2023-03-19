import os

# #############################################
# # BenchMark Bike
# #############################################
# ########### NYC ########### --closeness_len 13 --period_len 0 --trend_len 0
os.system("python STGCN.py --dataset Bike --city NYC --data_range 0.25 --train_data_length 91 --MergeIndex 3 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # os.system("python STGCN.py --dataset Bike --city NYC --data_range 0.5 --train_data_length 183 --MergeIndex 6 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# os.system("python STGCN.py --dataset Bike --city NYC --data_range all --train_data_length 365 --MergeIndex 12 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")


# # # ########### Chicago ###########
# # # # os.system("python STGCN.py --dataset Bike --city Chicago --data_range 0.25 --train_data_length 91 --MergeIndex 3 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # # # os.system("python STGCN.py --dataset Bike --city Chicago --data_range 0.5 --train_data_length 183 --MergeIndex 6 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# os.system("python STGCN.py --dataset Bike --city Chicago --data_range all --train_data_length 365 --MergeIndex 12 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")


# # # ########### DC ###########
# # # # os.system("python STGCN.py --dataset Bike --city DC --data_range 0.25 --train_data_length 91 --MergeIndex 3 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # # # os.system("python STGCN.py --dataset Bike --city DC --data_range 0.5 --train_data_length 183 --MergeIndex 6 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# os.system("python STGCN.py --dataset Bike --city DC --data_range all --train_data_length 365 --MergeIndex 12 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")



# # # ###############################################
# # # # BenchMark DiDi
# # # ###############################################
# # # ############# Xian #############
# # # # os.system("python STGCN.py --dataset DiDi --city Xian --MergeIndex 3 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # # # os.system("python STGCN.py --dataset DiDi --city Xian --MergeIndex 6 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # os.system("python STGCN.py --dataset DiDi --city Xian --MergeIndex 12 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # # # ############# Chengdu #############
# # # # # os.system("python STGCN.py --dataset DiDi --city Chengdu --MergeIndex 3 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # # # # os.system("python STGCN.py --dataset DiDi --city Chengdu --MergeIndex 6 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # os.system("python STGCN.py --dataset DiDi --city Chengdu --MergeIndex 12 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")



# # # # ###############################################
# # # # # BenchMark Metro
# # # # ###############################################
# # # # ############# Chongqing #############
# # # # # os.system("python STGCN.py --dataset Metro --city Chongqing --MergeIndex 3 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # # # # os.system("python STGCN.py --dataset Metro --city Chongqing --MergeIndex 6 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # os.system("python STGCN.py --dataset Metro --city Chongqing --MergeIndex 12 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")


# # # # ############# Shanghai #############
# # # # # os.system("python STGCN.py --dataset Metro --city Shanghai --MergeIndex 3 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # # # # os.system("python STGCN.py --dataset Metro --city Shanghai --MergeIndex 6 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # os.system("python STGCN.py --dataset Metro --city Shanghai --MergeIndex 12 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")



# # # # ###############################################
# # # # # BenchMark ChargeStation
# # # # ###############################################

# # # # # os.system("python STGCN.py --dataset ChargeStation --city Beijing --MergeIndex 1 --MergeWay max --closeness_len 13 --period_len 0 --trend_len 0")

# # os.system("python STGCN.py --dataset ChargeStation --city Beijing --MergeIndex 2 --MergeWay max --closeness_len 13 --period_len 0 --trend_len 0")



# # # # ###############################################
# # # # # BenchMark METR-LA
# # # # ###############################################

# # # # # os.system("python STGCN.py --dataset METR --city LA --MergeIndex 3 --MergeWay average --closeness_len 13 --period_len 0 --trend_len 0")

# # # # # os.system("python STGCN.py --dataset METR --city LA --MergeIndex 6 --MergeWay average --closeness_len 13 --period_len 0 --trend_len 0")

# # os.system("python STGCN.py --dataset METR --city LA --MergeIndex 12 --MergeWay average --closeness_len 13 --period_len 0 --trend_len 0")


# # # # ###############################################
# # # # # BenchMark PEMS-BAY
# # # # ###############################################
# # # # # os.system("python STGCN.py --dataset PEMS --city BAY --MergeIndex 3 --MergeWay average --closeness_len 13 --period_len 0 --trend_len 0")

# # # # # os.system("python STGCN.py --dataset PEMS --city BAY --MergeIndex 6 --MergeWay average --closeness_len 13 --period_len 0 --trend_len 0")

# # os.system("python STGCN.py --dataset PEMS --city BAY --MergeIndex 12 --MergeWay average --closeness_len 13 --period_len 0 --trend_len 0")



# # # ############# Xian_Street #############
# os.system("python STGCN.py --dataset DiDi --city Xian_Street --MergeIndex 3 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# os.system("python STGCN.py --dataset DiDi --city Xian_Street --MergeIndex 6 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# os.system("python STGCN.py --dataset DiDi --city Xian_Street --MergeIndex 12 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# # # ############# Chengdu_Street #############
# os.system("python STGCN.py --dataset DiDi --city Chengdu_Street --MergeIndex 3 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# os.system("python STGCN.py --dataset DiDi --city Chengdu_Street --MergeIndex 6 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

# os.system("python STGCN.py --dataset DiDi --city Chengdu_Street --MergeIndex 12 --MergeWay sum --closeness_len 13 --period_len 0 --trend_len 0")

