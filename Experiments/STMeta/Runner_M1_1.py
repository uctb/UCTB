import os

############################################################################################################
# Enrich gcn_k
############################################################################################################

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:64,mark:BM211')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:64,mark:BM211')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:128,mark:BM211')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:128,mark:BM211')