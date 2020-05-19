import os
###############################################
# C P T trial
###############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance,MergeIndex:1,closeness_len:12,period_len:14,trend_len:8')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance,MergeIndex:2,closeness_len:12,period_len:14,trend_len:8')