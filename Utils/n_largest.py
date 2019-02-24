
def insertByOrderDEC(valueList, valueAndIndex, n, key=lambda x:x):
    index, value = valueAndIndex
    if len(valueList) == 0:
        valueList.append(valueAndIndex)
    else:
        for i in range(len(valueList)):
            if key(value) > key(valueList[i][1]):
                valueList.insert(i, valueAndIndex)
                break
            elif len(valueList) < n and i == len(valueList) - 1:
                valueList.append(valueAndIndex)
    return valueList[:n]

def n_largest(n, valueList, key=lambda x:x):
    resultList = []
    for i in range(len(valueList)):
        resultList = insertByOrderDEC(resultList, (i, valueList[i]), n, key=key)
    return resultList

