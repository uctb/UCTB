import os

#############################################
# BenchMark Bike
#############################################

# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
#           ' -p data_range:0.25,train_data_length:91,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')

# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml -p data_range:0.25,train_data_length:91,graph:Distance,MergeIndex:3')
# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml -p graph:Distance,MergeIndex:6')
# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml -p graph:Distance,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
          '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
          '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')


########### Chicago ###########
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml'
          ' -p data_range:0.25,train_data_length:91,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml'
          ' -p data_range:0.5,train_data_length:183,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml -p data_range:0.25,train_data_length:91,graph:Distance,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml -p data_range:0.5,train_data_length:183,graph:Distance,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml -p graph:Distance,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_chicago.data.yml '
          '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_chicago.data.yml '
          '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_chicago.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_chicago.data.yml '
          '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_chicago.data.yml '
          '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_chicago.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
          '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
          '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

############# DC #############
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml'
          ' -p data_range:0.25,train_data_length:91,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml'
          ' -p data_range:0.5,train_data_length:183,graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml -p data_range:0.25,train_data_length:91,graph:Distance,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml -p data_range:0.5,train_data_length:183,graph:Distance,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml -p graph:Distance,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_dc.data.yml '
          '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_dc.data.yml '
          '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_dc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_dc.data.yml '
          '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_dc.data.yml '
          '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_dc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
          '-p data_range:0.25,train_data_length:91,graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
          '-p data_range:0.5,train_data_length:183,graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')



###############################################
# BenchMark DiDi
###############################################

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







os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml -p graph:Distance,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml -p graph:Distance,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml -p graph:Distance,MergeIndex:12')


os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')


os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')


os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')


###############################################
# BenchMark Metro
###############################################

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


###############################################
# BenchMark ChargeStation
###############################################

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


###############################################
# C P T trial
###############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance,MergeIndex:1,closeness_len:12,period_len:14,trend_len:8')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance,MergeIndex:2,closeness_len:12,period_len:14,trend_len:8')