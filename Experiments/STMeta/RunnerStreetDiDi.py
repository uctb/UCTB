import os

###############################################
# BenchMark DiDi
###############################################
############# Xian #############
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian_street.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian_street.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian_street.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian_street.data.yml -p graph:Distance,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian_street.data.yml -p graph:Distance,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian_street.data.yml -p graph:Distance,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_xian_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

############# Chengdu #############
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu_street.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu_street.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu_street.data.yml'
          ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu_street.data.yml -p graph:Distance,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu_street.data.yml -p graph:Distance,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu_street.data.yml -p graph:Distance,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_chengdu_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:3')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:6')
os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d didi_chengdu_street.data.yml '
          '-p graph:Distance-Correlation-Interaction,MergeIndex:12')

