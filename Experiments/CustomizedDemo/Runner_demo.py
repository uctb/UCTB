import os
import numpy as np
import heapq

#############################################
# BenchMark Metro(demo)
#############################################

os.system('python STMeta_Obj_Cus.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml '
          '-p graph:Distance-Correlation-Line-TopK,MergeIndex:3')
# os.system('python STMeta_Obj_Cus.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line-TopK,MergeIndex:3')
# os.system('python STMeta_Obj_Cus.py -m STMeta_v2.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line-TopK,MergeIndex:3')
# os.system('python STMeta_Obj_Cus.py -m STMeta_v3.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line-TopK,MergeIndex:3')
