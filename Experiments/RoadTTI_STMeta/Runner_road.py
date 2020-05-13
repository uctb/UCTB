import os
from tqdm import tqdm

#############################################
# BenchMark Chengdu Dataset
#############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Chengdu.data.yml'
          ' -p period_len:0,trend_len:0,graph:Road_Distance,st_method:LSTM,mark:LSTMC')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Chengdu.data.yml -p graph:Road_Distance')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d DiDiTTI_Chengdu.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d DiDiTTI_Chengdu.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d DiDiTTI_Chengdu.data.yml -p graph:Road_distance-Correlation')


#############################################
# BenchMark Haikou Dataset
#############################################
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Haikou.data.yml'
          ' -p period_len:0,trend_len:0,graph:Road_Distance,st_method:LSTM,mark:LSTMC')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Haikou.data.yml -p graph:Road_Distance')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d DiDiTTI_Haikou.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d DiDiTTI_Haikou.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d DiDiTTI_Haikou.data.yml -p graph:Road_distance-Correlation')


#############################################
# BenchMark Jinan Dataset
#############################################
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Jinan.data.yml'
          ' -p period_len:0,trend_len:0,graph:Road_Distance,st_method:LSTM,mark:LSTMC')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Jinan.data.yml -p graph:Road_Distance')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d DiDiTTI_Jinan.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d DiDiTTI_Jinan.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d DiDiTTI_Jinan.data.yml -p graph:Road_distance-Correlation')



#############################################
# BenchMark Shenzhen Dataset
#############################################
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Shenzhen.data.yml'
          ' -p period_len:0,trend_len:0,graph:Road_Distance,st_method:LSTM,mark:LSTMC')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Shenzhen.data.yml -p graph:Road_Distance')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d DiDiTTI_Shenzhen.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d DiDiTTI_Shenzhen.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d DiDiTTI_Shenzhen.data.yml -p graph:Road_distance-Correlation')


#############################################
# BenchMark Suzhou Dataset
#############################################
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Suzhou.data.yml'
          ' -p period_len:0,trend_len:0,graph:Road_Distance,st_method:LSTM,mark:LSTMC')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Suzhou.data.yml -p graph:Road_Distance')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d DiDiTTI_Suzhou.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d DiDiTTI_Suzhou.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d DiDiTTI_Suzhou.data.yml -p graph:Road_distance-Correlation')


#############################################
# BenchMark Xian Dataset
#############################################
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Xian.data.yml'
          ' -p period_len:0,trend_len:0,graph:Road_Distance,st_method:LSTM,mark:LSTMC')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d DiDiTTI_Xian.data.yml -p graph:Road_Distance')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d DiDiTTI_Xian.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d DiDiTTI_Xian.data.yml -p graph:Road_distance-Correlation')

os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d DiDiTTI_Xian.data.yml -p graph:Road_distance-Correlation')