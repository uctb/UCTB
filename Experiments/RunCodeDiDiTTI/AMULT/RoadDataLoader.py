import os
from UCTB.dataset import NodeTrafficLoader
import pickle
import numpy as np
from geopy.distance import distance
from tqdm import tqdm
from UCTB.model_unit import GraphBuilder


class RoadDataLoader(NodeTrafficLoader):

    def __init__(self, graph, with_lm=True, **kwargs):

        super(RoadDataLoader, self).__init__(graph=graph, with_lm=True, **kwargs)

        if with_lm:
            for graph_name in graph.split('-'):
                if graph_name.lower() == 'road_distance':
                    pass
                    ###################################################################
                    # build adjacency matrix
                    
                    adj_path = os.path.join(os.getcwd(),"{}_adjacency.tmp".format(city))
                    #adjacency matrix exists .load it 
                    if os.path.exists(adj_path):
                        with open(adj_path,"rb") as fptmp:
                            adjacency = pickle.load(fptmp)
                    else:
                        #adjacency matrix doesn't exist create first
                        extrLatLng = lambda x:x[2:8]
                        adjacency = PointlistToAdjacency([extrLatLng(x) for x in self.dataset.data['Node']['StationInfo']])
                        with open(adj_path,"wb") as fptmp:
                            pickle.dump(adjacency,fptmp)

                    ###################################################################
                    self.AM.append(adjacency)
                    #build laplace matrix
                    self.LM = np.vstack((self.LM,(GraphBuilder.adjacent_to_laplacian(self.AM[-1]))[np.newaxis,:]))



def PointlistToAdjacency(pointlist):
    '''
    param "pointlist" should be like [104.04135, 30.6447, 104.04082, 30.66078, 104.04265, 30.67822]
    lat,lng should exist in pair
    '''
    print("Calculate distance among points")
    pointNum = len(pointlist)

    Adjacency = np.zeros((pointNum, pointNum))

    for i in tqdm(range(pointNum)):
        ilen = len(pointlist[i])
        assert ilen % 2 == 0, "lat,lng doesn't exist in pair. please check pointlist"
        for j in range(i+1, pointNum):
            jlen = len(pointlist[j])
            assert jlen % 2 == 0, "lat,lng doesn't exist in pair. please check pointlist"

            dis_count = 0
            for i_index in range(0, ilen, 2):
                for j_index in range(0, jlen, 2):
                    dis_count += int(distance((pointlist[i][i_index+1], pointlist[i][i_index]),
                                              (pointlist[j][j_index+1], pointlist[j][j_index])).meters)

            dis_count //= (ilen*jlen/4)

            Adjacency[i, j] = int(dis_count)
            Adjacency[j, i] = Adjacency[i, j]

    return Adjacency

