StrictDataFormat = {
    "TimeRange": [],    # 起止时间 str eg:['2016-10-01', '2016-11-30']
    "TimeFitness": [],  # 时间粒度 int 单位为min
    "Node": {
        "TrafficNode": [],  # shape(1440,256)
        # shape(120, 256, 256)  with shape [month, num-of-node. num-of-node]
        "TrafficMonthlyInteraction": [],
        "StationInfo": [],  # len为256的list eg:['0', 0, 34.210542575000005, 108.91390095, 'grid_0']
                            # {id (may be arbitrary): [id (when sorted, should be consistant with index of node_traffic), latitude, longitude, other notes]}
        "POI": []
    },
    "Grid": {
        "TrafficGrid": [],  # (1440, 16, 16)
        "GridLatLng": [],  # len为17的list eg:[34.20829427, 108.91118]
        "POI": []
    },
    "ExternalFeature": {
        "Weather": []
    }
}
