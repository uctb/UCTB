import os

###############################################
# BenchMark DiDi

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d didi_xian.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml -p graph:Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
#
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d didi_chengdu.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml -p graph:Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation-Interaction')

###############################################
# BenchMark Metro

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d metro_chongqing.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml -p graph:Line')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Distance-Correlation-Line')
#
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d metro_shanghai.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_shanghai.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_shanghai.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_shanghai.data.yml -p graph:Line')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line')


###############################################
# Transfer Bike-Pretrain
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance,closeness_len:6,period_len:0,trend_len:0,train_data_length:120,mark:C6')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml '
          '-p graph:Distance,closeness_len:6,period_len:0,trend_len:0,train_data_length:120,mark:C6')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_dc.data.yml '
          '-p graph:Distance,closeness_len:6,period_len:0,trend_len:0,train_data_length:120,mark:C6')