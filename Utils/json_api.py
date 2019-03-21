import json


def getJson(fullPath, showMessage=False):
    with open(fullPath, 'r') as f:
        data = json.load(f)
    if showMessage:
        print('load', fullPath)
    return data


def saveJson(dataDict, fullPath, showMessage=False):
    with open(fullPath, 'w') as f:
        json.dump(dataDict, f)
    if showMessage:
        print('Saved', fullPath)

def getJsonDataFromPath(fullPath, showMessage=False):
    with open(fullPath, 'r') as f:
        data = json.load(f)
    if showMessage:
        print('load', fullPath)
    return data