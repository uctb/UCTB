import os

#############################################
# BenchMark Bike
#############################################
########### NYC ###########

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
#           '-p graph:Correlation,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
#           '-p graph:Interaction,MergeIndex:12')

# ########### Chicago ###########
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
#           '-p graph:Distance,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
#           '-p graph:Correlation,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
#           '-p graph:Interaction,MergeIndex:12')

# ############# DC #############
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
#           '-p graph:Distance,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
#           '-p graph:Correlation,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
          '-p graph:Interaction,MergeIndex:12')

###############################################
# BenchMark DiDi
###############################################
############# Xian #############
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
#           '-p graph:Distance,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
#           '-p graph:Correlation,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
#           '-p graph:Interaction,MergeIndex:12')

# # ############# Chengdu #############
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Correlation,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Interaction,MergeIndex:12')

###############################################
# BenchMark Metro
###############################################
############# Chongqing #############
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Distance,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Correlation,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Line,MergeIndex:12')

# # ############# Shanghai #############
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Correlation,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Line,MergeIndex:12')

# ###############################################
# # BenchMark ChargeStation
# ###############################################
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance,MergeIndex:2')

# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Correlation,MergeIndex:2')

###############################################
# BenchMark METR-LA
###############################################

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
          ' -d metr_la.data.yml -p graph:Distance,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
          ' -d metr_la.data.yml -p graph:Correlation,MergeIndex:12')

###############################################
# BenchMark PEMS-BAY
###############################################

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
          ' -d pems_bay.data.yml -p graph:Distance,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
          ' -d pems_bay.data.yml -p graph:Correlation,MergeIndex:12')