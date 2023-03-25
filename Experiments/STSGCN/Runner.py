import os
# #############################################
# # BenchMark Bike
# #############################################
# ########### NYC ########### --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json
os.system("python STSGCN.py --dataset Bike --city NYC --data_range 0.25 --train_data_length 91 --MergeIndex 3 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ././config/PEMS03/STMeta_emb.json")

# # os.system("python STSGCN.py --dataset Bike --city NYC --data_range 0.5 --train_data_length 183 --MergeIndex 6 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ././config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset Bike --city NYC --data_range all --train_data_length 365 --MergeIndex 12 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ././config/PEMS03/STMeta_emb.json")


# # # ########### Chicago ###########
# # # # os.system("python STSGCN.py --dataset Bike --city Chicago --data_range 0.25 --train_data_length 91 --MergeIndex 3 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ././config/PEMS03/STMeta_emb.json")

# # # # os.system("python STSGCN.py --dataset Bike --city Chicago --data_range 0.5 --train_data_length 183 --MergeIndex 6 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset Bike --city Chicago --data_range all --train_data_length 365 --MergeIndex 12 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")


# # # ########### DC ###########
# # # # os.system("python STSGCN.py --dataset Bike --city DC --data_range 0.25 --train_data_length 91 --MergeIndex 3 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# # # # os.system("python STSGCN.py --dataset Bike --city DC --data_range 0.5 --train_data_length 183 --MergeIndex 6 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset Bike --city DC --data_range all --train_data_length 365 --MergeIndex 12 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")



# # # ###############################################
# # # # BenchMark DiDi
# # # ###############################################
# # # ############# Xian #############
# # # # os.system("python STSGCN.py --dataset DiDi --city Xian --MergeIndex 3 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# # # # os.system("python STSGCN.py --dataset DiDi --city Xian --MergeIndex 6 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset DiDi --city Xian --MergeIndex 12 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")



# # # # ############# Chengdu #############
# # # # # os.system("python STSGCN.py --dataset DiDi --city Chengdu --MergeIndex 3 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# # # # # os.system("python STSGCN.py --dataset DiDi --city Chengdu --MergeIndex 6 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset DiDi --city Chengdu --MergeIndex 12 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config config/PEMS03/STMeta_emb_1.json")



# # # # ###############################################
# # # # # BenchMark Metro
# # # # ###############################################
# # # # ############# Chongqing #############
# # # # # os.system("python STSGCN.py --dataset Metro --city Chongqing --MergeIndex 3 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# # # # # os.system("python STSGCN.py --dataset Metro --city Chongqing --MergeIndex 6 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset Metro --city Chongqing --MergeIndex 12 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")


# # # # ############# Shanghai #############
# # # # # os.system("python STSGCN.py --dataset Metro --city Shanghai --MergeIndex 3 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# # # # # os.system("python STSGCN.py --dataset Metro --city Shanghai --MergeIndex 6 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset Metro --city Shanghai --MergeIndex 12 --MergeWay sum --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb_1.json")



# # ###############################################
# # # BenchMark ChargeStation
# # ###############################################

# # # os.system("python STSGCN.py --dataset ChargeStation --city Beijing --MergeIndex 1 --MergeWay max --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset ChargeStation --city Beijing --MergeIndex 2 --MergeWay max --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")



# # # ###############################################
# # # # BenchMark METR-LA
# # # ###############################################

# # # # os.system("python STSGCN.py --dataset METR --city LA --MergeIndex 3 --MergeWay average --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# # # # os.system("python STSGCN.py --dataset METR --city LA --MergeIndex 6 --MergeWay average --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset METR --city LA --MergeIndex 12 --MergeWay average --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")


# # ###############################################
# # # BenchMark PEMS-BAY
# # ###############################################
# # # os.system("python STSGCN.py --dataset PEMS --city BAY --MergeIndex 3 --MergeWay average --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# # # os.system("python STSGCN.py --dataset PEMS --city BAY --MergeIndex 6 --MergeWay average --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")

# os.system("python STSGCN.py --dataset PEMS --city BAY --MergeIndex 12 --MergeWay average --closeness_len 12 --period_len 0 --trend_len 0 --config ./config/PEMS03/STMeta_emb.json")


