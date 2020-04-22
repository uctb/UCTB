import os

#############################################
# BenchMark Bike
#############################################

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p external_method:not,graph:Distance-Correlation-Interaction,mark:not_external_feature_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p external_method:direct,graph:Distance-Correlation-Interaction,mark:direct_concat_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p external_method:embedding,graph:Distance-Correlation-Interaction,mark:one_embedding_layer_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p external_method:classified,graph:Distance-Correlation-Interaction,mark:classified_embedding_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d bike_nyc.data.yml '
          '-p external_method:gating,graph:Distance-Correlation-Interaction,mark:gating_fusion_V1')



# ###############################################
# # BenchMark DiDi
# ###############################################

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p external_method:not,graph:Distance-Correlation-Interaction,mark:not_external_feature_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p external_method:direct,graph:Distance-Correlation-Interaction,mark:direct_concat_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p external_method:embedding,graph:Distance-Correlation-Interaction,mark:one_embedding_layer_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p external_method:classified,graph:Distance-Correlation-Interaction,mark:classified_embedding_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p external_method:gating,graph:Distance-Correlation-Interaction,mark:gating_fusion_V1')



# ###############################################
# # BenchMark Metro
# ###############################################

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
          '-p external_method:not,graph:Distance-Correlation-Line,mark:not_external_feature_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
          '-p external_method:direct,graph:Distance-Correlation-Line,mark:direct_concat_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
          '-p external_method:embedding,graph:Distance-Correlation-Line,mark:one_embedding_layer_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
          '-p external_method:classified,graph:Distance-Correlation-Line,mark:classified_embedding_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d metro_shanghai.data.yml '
          '-p external_method:gating,graph:Distance-Correlation-Line,mark:gating_fusion_V1')


# ###############################################
# # BenchMark ChargeStation
# ###############################################

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d chargestation_beijing.data.yml '
          '-p external_method:not,graph:Distance-Correlation,mark:not_external_feature_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d chargestation_beijing.data.yml '
          '-p external_method:direct,graph:Distance-Correlation,mark:direct_concat_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d chargestation_beijing.data.yml '
          '-p external_method:embedding,graph:Distance-Correlation,mark:one_embedding_layer_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d chargestation_beijing.data.yml '
          '-p external_method:classified,graph:Distance-Correlation,mark:classified_embedding_V1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d chargestation_beijing.data.yml '
          '-p external_method:gating,graph:Distance-Correlation,mark:gating_fusion_V1')

