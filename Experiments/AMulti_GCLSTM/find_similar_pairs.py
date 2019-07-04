from UCTB.dataset import NodeTrafficLoader
from UCTB.preprocess.time_utils import is_work_day_chine, is_work_day_america

source_dataset = 'Bike'
source_city = 'DC'

target_dataset = 'Bike'
target_city = 'NYC'

target_data_length = 1  # day

source_data_loader = NodeTrafficLoader(dataset=source_dataset, city=source_city,
                                       closeness_len=6, period_len=7, trend_len=4,
                                       normalize=True, with_lm=False, with_tpe=False,
                                       workday_parser=is_work_day_america)

target_data_loader = NodeTrafficLoader(dataset=target_dataset, city=target_city,
                                       train_data_length=target_data_length,
                                       closeness_len=6, period_len=7, trend_len=4,
                                       normalize=True, with_lm=False, with_tpe=False,
                                       workday_parser=is_work_day_america)

print('debug')