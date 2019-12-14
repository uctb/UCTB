# contain some method for processing DiDi TTI dataset.
import numpy as np
import re

def FixRow(row):
    zero = np.argwhere(row == 0).flatten()
    for index in zero:

        offset_ind = 1008
        while(True):
            if((index+offset_ind) < row.shape[0] and row[index+offset_ind] != 0):
                row[index] = row[index+offset_ind]
                break
            if((index-offset_ind) >= 0 and row[index-offset_ind] != 0):
                row[index] = row[index-offset_ind]
                break
            if((index+offset_ind) >= row.shape[0] and (index-offset_ind) < 0):
                break

            offset_ind += 1008


def FixRowByDay(row):
    zero = np.argwhere(row == 0).flatten()
    for index in zero:

        offset_ind = 144
        while(True):
            if((index+offset_ind) < row.shape[0] and row[index+offset_ind] != 0):
                row[index] = row[index+offset_ind]
                break
            if((index-offset_ind) >= 0 and row[index-offset_ind] != 0):
                row[index] = row[index-offset_ind]
                break
            if((index+offset_ind) >= row.shape[0] and (index-offset_ind) < 0):
                break

            offset_ind += 144


def FixMatrix(M):
    _, col = M.shape
    for c_ind in range(col):
        # when the numpy array is arugument ,it's shallow copy,namely it is a pointer
        FixRow(M[:, c_ind])


def FixMatrixByDay(M):
    _, col = M.shape
    for c_ind in range(col):
        # when the numpy array is arugument ,it's shallow copy,namely it is a pointer
        FixRowByDay(M[:, c_ind])


def str2point(strinfo):
    float_pattern = re.compile(r'\d+\.\d+ \d+\.\d+')
    tmp = re.findall(float_pattern,strinfo)
    point = []
    for row in tmp:
        try:
            locInfo = row.split()
            point.append((float(locInfo[0]),float(locInfo[1])))
        except Exception as e:
            print(e,row)

    bylng = sorted(point,key=lambda x:x[0])
    bylat = sorted(point,key=lambda x:x[1])

    lng_delta = bylng[-1][0]-bylng[0][0]
    lat_delta = bylat[-1][1]-bylat[0][1]
    if lng_delta > lat_delta:
        return [bylng[0],bylng[len(bylng)//2],bylng[-1]]
    else:
        return [bylat[0],bylat[len(bylat)//2],bylat[-1]]