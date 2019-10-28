import os

#############################################
# BenchMark Bike
#############################################

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d bike_nyc.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d bike_chicago.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d bike_dc.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_nyc.data.yml -p '
#           'graph:Correlation,gcn_k:2,mark:BenchMark2,gclstm_layers:2')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_nyc.data.yml -p '
#           'graph:Correlation,gcn_k:3,mark:BenchMark3,gclstm_layers:3')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d bike_nyc.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_nyc.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_nyc.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_nyc.data.yml -p graph:Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d bike_chicago.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml -p graph:Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_chicago.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d bike_chicago.data.yml '
#           '-p graph:Distance-Correlation-Interaction')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d bike_dc.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_dc.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_dc.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_dc.data.yml -p graph:Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_dc.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d bike_dc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d bike_dc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')


# ###############################################
# # BenchMark DiDi
# ###############################################
#
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d didi_xian.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d didi_chengdu.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d didi_xian.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml -p graph:Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d didi_xian.data.yml '
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
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation-Interaction')

###############################################
# BenchMark Metro
###############################################

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d metro_chongqing.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d metro_shanghai.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d metro_chongqing.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml -p graph:Line')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Distance-Correlation-Line')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d metro_chongqing.data.yml '
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
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,gclstm_layers:1,gcn_k:3,mark:BM13')

###############################################
# BenchMark ChargeStation
###############################################

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d chargestation_beijing.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml'
#             ' -d chargestation_beijing.data.yml -p graph:Distance')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Correlation')
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')

# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v2.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
#
# os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v3.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
