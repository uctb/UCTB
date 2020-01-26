import os

#############################################
# BenchMark Bike
#############################################

# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:NotEmbeddinng_V0')
# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,embedding_flag:True,mark:OneLayerEmbedding_V0')    
# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,embedding_flag:True,classified_embedding:True,mark:ClassifiedEmbedding_V0')


# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance-Correlation-Interaction,mark:NotEmbeddinng_V1')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance-Correlation-Interaction,embedding_flag:True,mark:OneLayerEmbeddingV1')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml'
#           '-p graph:Distance-Correlation-Interaction,embedding_flag:True,classified_embedding:True,mark:ClassifiedEmbedding_V1')


# os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
# os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_chicago.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
# os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d bike_dc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')


# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_nyc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_chicago.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml -d bike_dc.data.yml '
#           '-p graph:Distance-Correlation-Interaction')


# ###############################################
# # BenchMark DiDi
# ###############################################
#
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,mark:NotEmbeddinng_V1')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,embedding_flag:True,mark:OneLayerEmbeddingV1')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction,embedding_flag:True,classified_embedding:True,mark:ClassifiedEmbedding_V1')



# os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d didi_xian.data.yml '
#           '-p graph:Distance-Correlation-Interaction')
# #
# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml -p graph:Distance')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml -p graph:Distance')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml -p graph:Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml -p graph:Interaction')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation-Interaction,gcn_k:1,gclstm_layers:1,batch_size:16,mark:K11')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation-Interaction,gcn_k:1,gclstm_layers:2,batch_size:32,mark:K12')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation-Interaction,gcn_k:1,gclstm_layers:3,batch_size:32,mark:K13')

# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation-Interaction,gcn_k:2,gclstm_layers:1,batch_size:16,mark:K21')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
#           '-p graph:Distance-Correlation-Interaction,gcn_k:3,gclstm_layers:1,batch_size:16,mark:K31')

# ###############################################
# # BenchMark Metro
# ###############################################

# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')
# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')

# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml -p graph:Distance')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml -p graph:Distance')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml -p graph:Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml -p graph:Line')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Distance-Correlation-Line')
# os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_chongqing.data.yml '
#           '-p graph:Distance-Correlation-Line')

# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml -p graph:Distance')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p graph:Distance')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p graph:Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml -p graph:Line')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line')
# os.system('python STMeta_Obj.py -m STMeta_v2.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line')

# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,gclstm_layers:1,gcn_k:3,mark:BM13')

# ###############################################
# # BenchMark ChargeStation
# ###############################################

# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d chargestation_beijing.data.yml'
#           ' -p graph:Distance,period_len:0,trend_len:0,mark:LSTMC')

# os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
#             ' -d chargestation_beijing.data.yml -p graph:Distance')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')

# os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
