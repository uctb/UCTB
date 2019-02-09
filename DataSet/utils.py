import datetime
from dateutil.parser import parse

publicHolidayList = ['01-01', '01-02', '01-16', '02-12', '02-13', '02-20', '05-29', '07-04', '09-04',
                     '10-09', '11-10', '11-11', '11-23', '12-25']


def is_work_day(date):
    if type(date) is str:
        date = parse(date)
    if date.strftime('%m-%d') in publicHolidayList:
        return False
    week = date.weekday()
    if week < 5:
        return True
    else:
        return False

