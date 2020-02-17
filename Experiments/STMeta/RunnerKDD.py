import os

#############################################
# BenchMark Bike
#############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml -p graph:Distance,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml -p graph:Distance,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml -p graph:Distance,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

# ###############################################
# # BenchMark DiDi
# ###############################################
#
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml -p graph:Distance,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml -p graph:Distance,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml -p graph:Distance,MergeIndex:12')


os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')


os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')


os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

# ###############################################
# # BenchMark Metro
# ###############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml -p graph:Distance,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml -p graph:Distance,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml -p graph:Distance,MergeIndex:12')


os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance-Correlation-Line,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance-Correlation-Line,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance-Correlation-Line,MergeIndex:12')


os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance-Correlation-Line,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance-Correlation-Line,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance-Correlation-Line,MergeIndex:12')    

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance-Correlation-Line,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance-Correlation-Line,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance-Correlation-Line,MergeIndex:12')


# ###############################################
# # BenchMark ChargeStation
# ###############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d chargestation_beijing.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:1')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d chargestation_beijing.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:2')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance,MergeIndex:1')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance,MergeIndex:2')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:1')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:2')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:1')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:2')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation,MergeIndex:2')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
