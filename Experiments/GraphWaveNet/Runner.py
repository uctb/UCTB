import os

#############################################
# BenchMark Bike
#############################################
########### NYC ###########
os.system("python GraphWaveNet.py --dataset Bike --city NYC --data_range 0.25 --train_data_length 91 --MergeIndex 3 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset Bike --city NYC --data_range 0.5 --train_data_length 183 --MergeIndex 6 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset Bike --city NYC --data_range all --train_data_length 365 --MergeIndex 12 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")


# ########### Chicago ###########
# # os.system("python GraphWaveNet.py --dataset Bike --city Chicago --data_range 0.25 --train_data_length 91 --MergeIndex 3 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# # os.system("python GraphWaveNet.py --dataset Bike --city Chicago --data_range 0.5 --train_data_length 183 --MergeIndex 6 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset Bike --city Chicago --data_range all --train_data_length 365 --MergeIndex 12 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")


# ########### DC ###########
# # os.system("python GraphWaveNet.py --dataset Bike --city DC --data_range 0.25 --train_data_length 91 --MergeIndex 3 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# # os.system("python GraphWaveNet.py --dataset Bike --city DC --data_range 0.5 --train_data_length 183 --MergeIndex 6 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset Bike --city DC --data_range all --train_data_length 365 --MergeIndex 12 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")



# ###############################################
# # BenchMark DiDi
# ###############################################
# ############# Xian #############
# # os.system("python GraphWaveNet.py --dataset DiDi --city Xian --MergeIndex 3 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# # os.system("python GraphWaveNet.py --dataset DiDi --city Xian --MergeIndex 6 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset DiDi --city Xian --MergeIndex 12 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# ############# Chengdu #############
# # os.system("python GraphWaveNet.py --dataset DiDi --city Chengdu --MergeIndex 3 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# # os.system("python GraphWaveNet.py --dataset DiDi --city Chengdu --MergeIndex 6 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset DiDi --city Chengdu --MergeIndex 12 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")



# ###############################################
# # BenchMark Metro
# ###############################################
# ############# Chongqing #############
# # os.system("python GraphWaveNet.py --dataset Metro --city Chongqing --MergeIndex 3 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# # os.system("python GraphWaveNet.py --dataset Metro --city Chongqing --MergeIndex 6 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset Metro --city Chongqing --MergeIndex 12 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")


# ############# Shanghai #############
# # os.system("python GraphWaveNet.py --dataset Metro --city Shanghai --MergeIndex 3 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# # os.system("python GraphWaveNet.py --dataset Metro --city Shanghai --MergeIndex 6 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset Metro --city Shanghai --MergeIndex 12 --MergeWay sum --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")



# ###############################################
# # BenchMark ChargeStation
# ###############################################

# os.system("python GraphWaveNet.py --dataset ChargeStation --city Beijing --MergeIndex 1 --MergeWay max --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset ChargeStation --city Beijing --MergeIndex 2 --MergeWay max --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")


# ###############################################
# # BenchMark METR-LA
# ###############################################

# # os.system("python GraphWaveNet.py --dataset METR --city LA --MergeIndex 3 --MergeWay average --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# # os.system("python GraphWaveNet.py --dataset METR --city LA --MergeIndex 6 --MergeWay average --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset METR --city LA --MergeIndex 12 --MergeWay average --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")


# ###############################################
# # BenchMark PEMS-BAY
# ###############################################
# # os.system("python GraphWaveNet.py --dataset PEMS --city BAY --MergeIndex 3 --MergeWay average --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# # os.system("python GraphWaveNet.py --dataset PEMS --city BAY --MergeIndex 6 --MergeWay average --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")

# os.system("python GraphWaveNet.py --dataset PEMS --city BAY --MergeIndex 12 --MergeWay average --gcn_bool --adjtype doubletransition --addaptadj  --randomadj")
