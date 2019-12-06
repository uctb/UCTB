import os

############################################################################################################
# Enrich gcn_k
############################################################################################################

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p '
          'gcn_k:1,gcn_layers:1,gclstm_layers:1,batch_size:16,mark:PS1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:16,mark:PS2')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p '
          'gcn_k:3,gcn_layers:1,gclstm_layers:1,batch_size:16,mark:PS3')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p '
          'gcn_k:1,gcn_layers:1,gclstm_layers:1,batch_size:32,mark:PS4')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:32,mark:PS5')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p '
          'gcn_k:3,gcn_layers:1,gclstm_layers:1,batch_size:32,mark:PS6')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p '
          'gcn_k:1,gcn_layers:1,gclstm_layers:1,batch_size:64,mark:PS7')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:64,mark:PS8')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p '
          'gcn_k:3,gcn_layers:1,gclstm_layers:1,batch_size:64,mark:PS9')