import os

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d didi_xian.data.yml -p graph:Distance')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml -p graph:Distance')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml -p graph:Correlation')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml -p graph:Interaction')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction')

os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v0.model.yml -d didi_chengdu.data.yml -p graph:Distance')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml -p graph:Distance')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml -p graph:Correlation')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml -p graph:Interaction')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation')
os.system('python AMulti_GCLSTM_Obj.py -m amulti_gclstm_v1.model.yml -d didi_chengdu.data.yml '
          '-p graph:Distance-Correlation-Interaction')