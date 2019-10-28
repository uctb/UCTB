import os

#############################################
# BenchMark Bike
#############################################

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_GRU.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
#
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_GRU.model.yml -d bike_chicago.data.yml '
#           '-p graph:Distance-Correlation-Interaction')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_GRU.model.yml -d bike_dc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')


# ###############################################
# # BenchMark DiDi
# ###############################################

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_GRU.model.yml -d didi_xian.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
#
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_GRU.model.yml -d didi_chengdu.data.yml '
#           '-p mark:V0')

###############################################
# BenchMark Metro
###############################################

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_GRU.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance')
#
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_GRU.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line')

###############################################
# BenchMark ChargeStation
###############################################

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_GRU.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance,mark:V0')
