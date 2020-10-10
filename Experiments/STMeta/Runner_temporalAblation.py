import os

#############################################
# BenchMark Bike
#############################################
########### NYC ###########
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
          '-p period_len:0,trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
          '-p trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

########### Chicago ###########
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
          '-p period_len:0,trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
          '-p trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
#           '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

############# DC #############
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
          '-p period_len:0,trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
          '-p trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
#           '-p graph:Distance-Correlation-Interaction,MergeIndex:12')


###############################################
# BenchMark DiDi
###############################################
############# Xian #############

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
          '-p period_len:0,trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
          '-p trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
#           '-p graph:Distance-Correlation-Interaction,MergeIndex:12')


############# Chengdu #############

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
          '-p period_len:0,trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
          '-p trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation-Interaction,MergeIndex:12')


###############################################
# BenchMark Metro
###############################################
############# Chongqing #############

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
          '-p period_len:0,trend_len:0,graph:Distance-Correlation-Line,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
          '-p trend_len:0,graph:Distance-Correlation-Line,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Distance-Correlation-Line,MergeIndex:12')

############# Shanghai #############

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
          '-p period_len:0,trend_len:0,graph:Distance-Correlation-Line,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
          '-p trend_len:0,graph:Distance-Correlation-Line,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,MergeIndex:12')


###############################################
# BenchMark ChargeStation
###############################################

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d chargestation_beijing.data.yml '
          ' -p period_len:0,trend_len:0,graph:Distance-Correlation,MergeIndex:2')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d chargestation_beijing.data.yml '
          ' -p trend_len:0,graph:Distance-Correlation,MergeIndex:2')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d chargestation_beijing.data.yml '
          ' -p graph:Distance-Correlation,MergeIndex:2')


###############################################
# BenchMark METR-LA
###############################################

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metr_la.data.yml '
          ' -p period_len:0,trend_len:0,graph:Distance-Correlation,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metr_la.data.yml '
          ' -p trend_len:0,graph:Distance-Correlation,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metr_la.data.yml '
          ' -p graph:Distance-Correlation,MergeIndex:12')


###############################################
# BenchMark PEMS-BAY
###############################################

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d pems_bay.data.yml'
          ' -p period_len:0,trend_len:0,graph:Distance-Correlation,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d pems_bay.data.yml'
          ' -p trend_len:0,graph:Distance-Correlation,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d pems_bay.data.yml'
          ' -p graph:Distance-Correlation,MergeIndex:12')
