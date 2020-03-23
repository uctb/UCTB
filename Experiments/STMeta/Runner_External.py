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



# ###############################################
# # BenchMark Metro
# ###############################################
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,mark:NoExternalfeature')


# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,mark:NotEmbeddinng_V1')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,embedding_flag:True,mark:OneLayerEmbeddingV1')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p lr:1e-4,graph:Distance-Correlation-Line,embedding_flag:True,mark:OneLayerEmbeddingV1lr4')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p lr:1e-4,graph:Distance-Correlation-Line,embedding_flag:True,classified_embedding:True,mark:ClassifiedEmbedding_V1lr4')


# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,mark:NotEmbeddinng_V1_lre6_400sgd')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,mark:NotEmbeddinng_V1_lre6_400')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,embedding_flag:True,mark:OneLayerEmbeddingV1_lre6_400')


# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
#           '-p graph:Distance-Correlation-Line,gclstm_layers:1,gcn_k:3,mark:BM13')

# ###############################################
# # BenchMark ChargeStation
# ###############################################


os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d chargestation_beijing.data.yml '
          '-p graph:Distance-Correlation,mark:NotEmbeddinng_V1')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d chargestation_beijing.data.yml '
          '-p graph:Distance-Correlation,embedding_flag:True,mark:OneLayerEmbeddingV1')
os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d chargestation_beijing.data.yml '
          '-p graph:Distance-Correlation,embedding_flag:True,classified_embedding:True,mark:ClassifiedEmbedding_V1')

# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v1.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')

# os.system('python STMeta_Obj.py -m STMeta_v2.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
# os.system('python STMeta_Obj.py -m STMeta_v3.model.yml'
#           ' -d chargestation_beijing.data.yml -p graph:Distance-Correlation')
