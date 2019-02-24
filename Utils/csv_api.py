# _*_ coding=utf-8 _*_
import csv

def loadCSVFileFromPath(fileName, fileWithHeader=False):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        if fileWithHeader:
            header = next(reader)
        else:
            header = []
        data = [r for r in reader]
    return header, data

def saveCSVFileToPath(fileName, fileHeader, dataList):
    with open(fileName, "w", newline='') as csvFile:
        writer = csv.writer(csvFile)
        # write header
        writer.writerow(fileHeader)
        # write content
        for record in dataList:
            writer.writerow(record)
    print('Saved', fileName)