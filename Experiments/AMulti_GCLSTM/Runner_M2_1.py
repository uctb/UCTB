import os

############################################################################################################
# Enrich gcn_k
############################################################################################################

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d chargestation_beijing.data.yml -p '
          'gcn_k:2,gcn_layers:1,gclstm_layers:1,batch_size:64,mark:BM211')