import os

############################################################################################################
# TMeta
############################################################################################################

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d bike_nyc.data.yml'
          ' -p graph:Distance,st_method:LSTM,model_version:LSTM,mark:V0')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d bike_chicago.data.yml'
          ' -p graph:Distance,st_method:LSTM,model_version:LSTM,mark:V0')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d bike_dc.data.yml'
          ' -p graph:Distance,st_method:LSTM,model_version:LSTM,mark:V0')