from scipy import stats


class EarlyStopping(object):
    def __init__(self, patience):
        self.__record_list = []
        self.__best = None
        self.__patience = patience
        self.__p = 0

    def stop(self, new_value):
        self.__record_list.append(new_value)
        if self.__best is None or new_value < self.__best:
            self.__best = new_value
            self.__p = 0
            return False
        else:
            if self.__p < self.__patience:
                self.__p += 1
                return False
            else:
                return True

class EarlyStoppingTTest(object):
    def __init__(self, length, p_value_threshold):
        self.__record_list = []
        self.__best = None
        self.__test_length = length
        self.__p_value_threshold = p_value_threshold

    def stop(self, new_value):
        self.__record_list.append(new_value)
        if len(self.__record_list) >= (self.__test_length * 2):
            lossTTest = stats.ttest_ind(self.__record_list[-self.__test_length:],
                                        self.__record_list[-self.__test_length * 2:-self.__test_length], equal_var=False)
            ttest = lossTTest[0]
            pValue = lossTTest[1]
            print('ttest:', ttest, 'pValue', pValue)
            if pValue > self.__p_value_threshold or ttest > 0:
                return True
            else:
                return False
        else:
            return False