import os

############################################################################################################
# Enrich gcn_k
############################################################################################################

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:1,gcn_layers:1,gclstm_layers:1,batch_size:16,mark:PS1')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:16,mark:PS2')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:3,gcn_layers:1,gclstm_layers:1,batch_size:16,mark:PS3')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:1,gcn_layers:1,gclstm_layers:1,batch_size:32,mark:PS4')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:32,mark:PS5')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:3,gcn_layers:1,gclstm_layers:1,batch_size:32,mark:PS6')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:1,gcn_layers:1,gclstm_layers:1,batch_size:64,mark:PS7')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:64,mark:PS8')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:3,gcn_layers:1,gclstm_layers:1,batch_size:64,mark:PS9')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:1,gcn_layers:1,gclstm_layers:1,batch_size:16,train_data_length:all,mark:PS10')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:16,train_data_length:all,mark:PS11')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:3,gcn_layers:1,gclstm_layers:1,batch_size:16,train_data_length:all,mark:PS12')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:1,gcn_layers:1,gclstm_layers:1,batch_size:32,train_data_length:all,mark:PS13')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:32,train_data_length:all,mark:PS14')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:3,gcn_layers:1,gclstm_layers:1,batch_size:32,train_data_length:all,mark:PS15')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:1,gcn_layers:1,gclstm_layers:1,batch_size:64,train_data_length:all,mark:PS16')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:64,train_data_length:all,mark:PS17')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p '
          'gcn_k:3,gcn_layers:1,gclstm_layers:1,batch_size:64,train_data_length:all,mark:PS18')