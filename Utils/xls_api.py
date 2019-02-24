# _*_ coding:utf-8 _*_
import xlrd

from localPath import *

def getDataFromXls(fileName, startRow=1, sheetName=None, sheetIndex=0):
    file = xlrd.open_workbook(fileName)
    if sheetName == None:
        fileSheet = file.sheet_by_index(sheetIndex)
    else:
        fileSheet = file.sheet_by_name(sheetName)

    fileData = []
    for i in range(startRow, fileSheet.nrows):
        fileData.append(fileSheet.row_values(i))

    return fileData