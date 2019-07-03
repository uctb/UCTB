import copy
import datetime
import numpy as np

from dateutil.parser import parse

from ..preprocess.time_utils import is_work_day_chine, is_work_day_america, is_valid_date
from ..preprocess import MoveSample, SplitData, ST_MoveSample, Normalizer
from ..model_unit import GraphBuilder

from .dataset import DataSet


class NodeTrafficLoader(object):
    
    def __init__(self,
                 dataset,
                 city,
                 data_range='All',
                 train_data_length='All',
                 test_ratio=0.1,
                 closeness_len=6,
                 period_len=4,
                 trend_len=7,
                 target_length=1,
                 graph='Correlation',
                 threshold_distance=1000,
                 threshold_correlation=0,
                 threshold_interaction=500,
                 normalize=False,
                 workday_parser=is_work_day_america,
                 with_lm=True,
                 with_tpe=False,
                 data_dir=None):

        self.dataset = DataSet(dataset, city, data_dir=data_dir)

        self.daily_slots = 24 * 60 / self.dataset.time_fitness

        if data_range.lower() == 'all':
            data_range = [0, len(self.dataset.node_traffic)]
        else:
            data_range = [int(e) for e in data_range.split(',')]
            data_range = [int(data_range[0] * self.daily_slots), int(data_range[1] * self.daily_slots)]

        num_time_slots = data_range[1] - data_range[0]

        # traffic feature
        traffic_data_index = np.where(np.mean(self.dataset.node_traffic, axis=0) * self.daily_slots > 1)[0]

        self.traffic_data = self.dataset.node_traffic[data_range[0]:data_range[1], traffic_data_index]

        # external feature
        external_feature = []
        # weather
        if len(self.dataset.external_feature_weather) > 0:
            external_feature.append(self.dataset.external_feature_weather[data_range[0]:data_range[1]])
        # Weekday Feature
        weekday_feature = [[1 if workday_parser(parse(self.dataset.time_range[1])
                                                + datetime.timedelta(hours=e * self.dataset.time_fitness / 60)) else 0] \
                           for e in range(data_range[0], num_time_slots + data_range[0])]
        # Hour Feature
        hour_feature = [[(parse(self.dataset.time_range[1]) +
                         datetime.timedelta(hours=e * self.dataset.time_fitness / 60)).hour / 24.0]
                        for e in range(data_range[0], num_time_slots + data_range[0])]
        
        external_feature.append(weekday_feature)
        external_feature.append(hour_feature)
        external_feature = np.concatenate(external_feature, axis=-1)

        time_embedding = copy.deepcopy(external_feature)
        
        self.station_number = self.traffic_data.shape[1]
        self.external_dim = external_feature.shape[1]
        
        if test_ratio > 1 or test_ratio < 0:
            raise ValueError('test_ratio ')
        train_test_ratio = [1 - test_ratio, test_ratio]

        self.train_data, self.test_data = SplitData.split_data(self.traffic_data, train_test_ratio)
        self.train_ef, self.test_ef = SplitData.split_data(external_feature, train_test_ratio)
        self.train_tpe, self.test_tpe = SplitData.split_data(time_embedding, train_test_ratio)

        # Normalize the traffic data
        if normalize:
            self.normalizer = Normalizer(self.train_data)
            self.train_data = self.normalizer.min_max_normal(self.train_data)
            self.test_data = self.normalizer.min_max_normal(self.test_data)

        if train_data_length.lower() != 'all':
            train_day_length = int(train_data_length)
            self.train_data = self.train_data[-int(train_day_length * self.daily_slots):]
            self.train_ef = self.train_ef[-int(train_day_length * self.daily_slots):]
            self.train_tpe = self.train_tpe[-int(train_day_length * self.daily_slots):]

        # expand the test data
        expand_start_index = len(self.train_data) -\
                             max(int(self.daily_slots * period_len),
                                 int(self.daily_slots * 7 * trend_len)) -\
                             closeness_len
        self.test_data = np.vstack([self.train_data[expand_start_index:], self.test_data])
        self.test_ef = np.vstack([self.train_ef[expand_start_index:], self.test_ef])
        self.test_tpe = np.vstack([self.train_tpe[expand_start_index:], self.test_tpe])

        # init move sample obj
        st_move_sample = ST_MoveSample(closeness_len=closeness_len,
                                       period_len=period_len,
                                       trend_len=trend_len, target_length=1, daily_slots=self.daily_slots)

        # Not finish yet
        # if with_tpe:
        #     self.traffic_data = np.concatenate([self.traffic_data, time_embedding], axis=-1)
        #
        # closeness, period, trend, y = st_move_sample.move_sample(self.traffic_data)

        self.train_closeness, \
        self.train_period, \
        self.train_trend, \
        self.train_y = st_move_sample.move_sample(self.train_data)

        self.test_closeness, \
        self.test_period, \
        self.test_trend, \
        self.test_y = st_move_sample.move_sample(self.test_data)

        # external feature
        self.train_ef = self.train_ef[-len(self.train_closeness) - target_length: -target_length]
        self.test_ef = self.test_ef[-len(self.test_closeness) - target_length: -target_length]

        if with_tpe:
            # Time position embedding
            self.train_closeness_tpe, \
            self.train_period_tpe, \
            self.train_trend_tpe, \
            _ = st_move_sample.move_sample(self.train_tpe)

            self.test_closeness_tpe, \
            self.test_period_tpe, \
            self.test_trend_tpe, \
            _ = st_move_sample.move_sample(self.test_tpe)

            self.train_closeness_tpe = np.tile(np.transpose(self.train_closeness_tpe, [0, 3, 2, 1]), [1, self.station_number, 1, 1])
            self.train_period_tpe = np.tile(np.transpose(self.train_period_tpe, [0, 3, 2, 1]), [1, self.station_number, 1, 1])
            self.train_trend_tpe = np.tile(np.transpose(self.train_trend_tpe, [0, 3, 2, 1]), [1, self.station_number, 1, 1])

            self.test_closeness_tpe = np.tile(np.transpose(self.test_closeness_tpe, [0, 3, 2, 1]), [1, self.station_number, 1, 1])
            self.test_period_tpe = np.tile(np.transpose(self.test_period_tpe, [0, 3, 2, 1]), [1, self.station_number, 1, 1])
            self.test_trend_tpe = np.tile(np.transpose(self.test_trend_tpe, [0, 3, 2, 1]), [1, self.station_number, 1, 1])

            # concat temporal feature with time position embedding
            self.train_closeness = np.concatenate((self.train_closeness, self.train_closeness_tpe, ), axis=-1)
            self.train_period = np.concatenate((self.train_period, self.train_period_tpe, ), axis=-1)
            self.train_trend = np.concatenate((self.train_trend, self.train_trend_tpe, ), axis=-1)

            self.test_closeness = np.concatenate((self.test_closeness, self.test_closeness_tpe,), axis=-1)
            self.test_period = np.concatenate((self.test_period, self.test_period_tpe,), axis=-1)
            self.test_trend = np.concatenate((self.test_trend, self.test_trend_tpe,), axis=-1)

        if with_lm:

            self.LM = []

            for graph_name in graph.split('-'):

                if graph_name.lower() == 'distance':

                    # Default order by date
                    order_parser = parse
                    try:
                        for key, value in self.dataset.node_station_info.items():
                            if is_valid_date(value[0]) is False:
                                order_parser = lambda x: x
                                print('Order by string')
                                break
                    except:
                        order_parser = lambda x: x
                        print('Order by string')

                    lat_lng_list = \
                        np.array([[float(e1) for e1 in e[1][1:3]] for e in
                                  sorted(self.dataset.node_station_info.items(),
                                         key=lambda x: order_parser(x[1][0]), reverse=False)])

                    self.LM.append(GraphBuilder.distance_graph(lat_lng_list=lat_lng_list[traffic_data_index],
                                                               threshold=float(threshold_distance)))

                if graph_name.lower() == 'interaction':
                    monthly_interaction = self.dataset.node_monthly_interaction[:, traffic_data_index, :][:, :,
                                          traffic_data_index]

                    monthly_interaction, _ = SplitData.split_data(monthly_interaction, train_test_ratio)

                    annually_interaction = np.sum(monthly_interaction[-12:-1], axis=0)
                    annually_interaction = annually_interaction + annually_interaction.transpose()

                    self.LM.append(GraphBuilder.interaction_graph(annually_interaction,
                                                                  threshold=float(threshold_interaction)))

                if graph_name.lower() == 'correlation':
                    self.LM.append(GraphBuilder.correlation_graph(self.train_data[-30 * int(self.daily_slots):],
                                                                  threshold=float(threshold_correlation),
                                                                  keep_weight=False))

                if graph_name.lower() == 'neighbor':
                    self.LM.append(
                        GraphBuilder.adjacent_to_lm(self.dataset.data.get('contribute_data').get('graph_neighbors')))

                if graph_name.lower() == 'line':
                    self.LM.append(GraphBuilder.adjacent_to_lm(self.dataset.data.get('contribute_data').get('graph_lines')))

                if graph_name.lower() == 'transfer':
                    self.LM.append(
                        GraphBuilder.adjacent_to_lm(self.dataset.data.get('contribute_data').get('graph_transfer')))

            self.LM = np.array(self.LM, dtype=np.float32)

    def st_map(self, zoom=11, style='light'):

        if self.dataset.node_station_info is None or len(self.dataset.node_station_info) == 0:
            raise ValueError('No station information found in dataset')

        import numpy as np
        import plotly
        from plotly.graph_objs import Scattermapbox, Layout

        mapboxAccessToken = "pk.eyJ1Ijoicm1ldGZjIiwiYSI6ImNqN2JjN3l3NjBxc3MycXAzNnh6M2oxbGoifQ.WFNVzFwNJ9ILp0Jxa03mCQ"

        order_parser = parse
        try:
            for key, value in self.dataset.node_station_info.items():
                if is_valid_date(value[0]) is False:
                    order_parser = lambda x: x
                    print('Order by string')
                    break
        except:
            order_parser = lambda x: x
            print('Order by string')

        lat_lng_name_list = [e[1][1:] for e in sorted(self.dataset.node_station_info.items(), key=lambda x: order_parser(x[1][0]), reverse=False)]
        build_order = list(range(len(lat_lng_name_list)))

        lng = [float(e[1]) for e in lat_lng_name_list]
        lat = [float(e[0]) for e in lat_lng_name_list]
        text = [e[-1] for e in lat_lng_name_list]

        file_name = self.dataset.dataset + '-' + self.dataset.city + '.html'

        bikeStations = [Scattermapbox(
            lon=lng,
            lat=lat,
            text=text,
            mode='markers',
            marker=dict(
                size=6,
                color=['rgb(%s, %s, %s)' % (255,
                                            195 - e * 195 / max(build_order),
                                            195 - e * 195 / max(build_order)) for e in build_order],
                opacity=1,
            ))]
        
        layout = Layout(
            title='Bike Station Location & The latest built stations with deeper color',
            autosize=True,
            hovermode='closest',
            showlegend=False,
            mapbox=dict(
                accesstoken=mapboxAccessToken,
                bearing=0,
                center=dict(
                    lat=np.median(lat),
                    lon=np.median(lng)
                ),
                pitch=0,
                zoom=zoom,
                style=style
            ),
        )

        fig = dict(data=bikeStations, layout=layout)
        plotly.offline.plot(fig, filename=file_name)