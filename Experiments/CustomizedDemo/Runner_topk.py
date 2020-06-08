import os
import numpy as np
import heapq

#############################################
# BenchMark Metro Shangahi (topK graph demo)
#############################################

os.system('python STMeta_Obj_topk.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml '
          '-p graph:TopK,MergeIndex:12')

os.system('python STMeta_Obj_topk.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
          '-p graph:Distance-Correlation-Line-TopK,MergeIndex:12')
          

