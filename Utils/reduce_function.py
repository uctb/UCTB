
def reduceByKey(valueList, key, func):
    if callable(key) and callable(func):
        resultDict = {}
        for element in valueList:
            if key(element) not in resultDict:
                resultDict[key(element)] = element
            else:
                resultDict[key(element)] = func(resultDict[key(element)], element)
        return [e[1] for e in resultDict.items()]

def reduce(valueList, func):
    if callable(func):
        result = valueList[0]
        for i in range(1, len(valueList)):
            result = func(result, valueList[i])
        return result