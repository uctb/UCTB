from abc import ABC, abstractmethod

class TemporalContextLoader(ABC):

    def __init__(self, traffic_dataloader):

        self.weather_data = self.get_weather()
        self.holiday_data = self.get_holiday()
        self.weather_data = self.get_temporal_position()
        pass

    @abstractmethod
    def get_weather(self, arg):
        '''
        Input:
            default weather features shape: (T, D_t)
            detailed weather features shape: (T, N, D_t)
            station info [[lat1, lng1], ....]

        Function to be implemented:
            Func 1 (T, ?, D_t) -> bind/replicate (optional) -> (T, N, D_t)
            Func 2 move Context sample
        '''
        # # func 1 推断天气
        # self.infer_weather()

        # func 2 构造历史context样本
        # # move Context sample
        # self.train_ef_closeness = None
        # self.train_ef_period = None
        # self.train_ef_trend = None
        # self.train_lstm_ef =  None
        # self.test_ef_closeness = None
        # self.test_ef_period = None
        # self.test_ef_trend = None
        # self.test_lstm_ef = None
        # if len(external_feature) > 0:
        #     self.external_move_sample = ST_MoveSample(closeness_len=self.closeness_len,
        #                                     period_len=self.period_len,
        #                                     trend_len=self.trend_len, target_length=0, daily_slots=self.daily_slots)

        #     self.train_ef_closeness, self.train_ef_period, self.train_ef_trend, _ = self.external_move_sample.move_sample(self.train_ef)

        #     self.test_ef_closeness, self.test_ef_period, self.test_ef_trend, _ = self.external_move_sample.move_sample(self.test_ef)


        #     if self.external_lstm_len is not None and self.external_lstm_len > 0:    
        #         self.external_move_sample = ST_MoveSample(closeness_len=self.external_lstm_len,period_len=0,trend_len=0, target_length=0, daily_slots=self.daily_slots)

        #         self.train_lstm_ef, _, _, _ = self.external_move_sample.move_sample(self.train_ef)

        #         self.test_lstm_ef, _, _, _ = self.external_move_sample.move_sample(self.test_ef)

        #     self.train_ef = self.train_ef[-self.train_sequence_len - target_length: -target_length]
        #     self.test_ef = self.test_ef[-self.test_sequence_len - target_length: -target_length]
            
        #     # weather
        #     self.train_lstm_ef = self.train_lstm_ef[-self.train_sequence_len - target_length: -target_length]
        #     self.test_lstm_ef = self.test_lstm_ef[-self.test_sequence_len - target_length: -target_length]
            
        # pass

    # @abstractmethod
    # def infer_weather(sellf, arg):
    #     pass
    
    @abstractmethod
    def get_holiday(sellf, arg):
        '''
        Input:
            parser function or csv file

        Function to be implemented:
            Func 1 (T//daily_slots, 1) -> temporal replicate ->  (T, 1)
        '''
        # parse by api

        # load by file
        pass
    
    # @staticmethod
    @abstractmethod
    def get_temporal_position(self, arg):
        ...



class SpatialContextLoader(ABC):
     
    def __init__(self, traffic_dataloader):
        self.poi = self.get_poi()
        
    @abstractmethod
    def get_poi(sellf, arg):
        '''
        Input:
            parser function or csv file

        Function to be implemented:
            Func 1 (?, N, D_s) -> temporal replicate -> (T, N, D_s)
        '''

        # load by file
        pass