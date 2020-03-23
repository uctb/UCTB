from dateutil.parser import parse
from chinese_calendar import is_workday
from workalendar.usa import NewYork, DistrictOfColumbia, Illinois

america_public_holiday = ['01-01', '01-02', '01-16', '02-12', '02-13', '02-20', '05-29', '07-04', '09-04',
                          '10-09', '11-10', '11-11', '11-23', '12-25']


def is_work_day_america(date, city):
    """
    Args:
        date(string or datetime): e.g. 2019-01-01

    Return:
        True if date is not holiday in America,
        otherwise return False.
    """
    if type(date) is str:
        date = parse(date)

    if city == "Chicago":
        workday = Illinois()
    elif city == "NYC":
        workday = NewYork()
    elif city == "DC":
        workday = DistrictOfColumbia()
    else:
        raise ValueError("can't parse holiday in {}.".format(city))
    return workday.is_working_day(date)


def is_work_day_china(date, city):
    """
    Args:
        date(string or datetime): e.g. 2019-01-01

    Return:
        True if date is not holiday in China,
        otherwise return False.
    """
    if type(date) is str:
        date = parse(date)
    return is_workday(date)


def is_valid_date(date_str):
    """
    Args:
        date_str(string): e.g. 2019-01-01

    Return:
        True if date_str is valid date,
        otherwise return False.
    """
    try:
        date = parse(date_str)
    except:
        return False

    year = date.year
    month = date.month
    day = date.day

    isRunNian = False
    if year % 4 == 0 and year % 100 != 0 and year % 400 == 0:
        isRunNian = True

    if month < 1 or month > 12:
        return False

    pingnian_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    runnian_month = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if isRunNian:
        if day < 1 or day > pingnian_month[month]:
            return False
    else:
        if day < 1 or day > runnian_month[month]:
            return False

    return True
