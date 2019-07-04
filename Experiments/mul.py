from multiprocessing import Pool, Manager
import os
from functools import reduce


# (my_rank, n_jobs, dataList, resultHandleFunction, parameterList)
def multipleProcess(distributeList, partitionDataFunc, taskFunction, n_jobs,
                    reduceFunction, parameterList):
    if callable(partitionDataFunc) and callable(taskFunction) and callable(reduceFunction):
        print('Parent process %s.' % os.getpid())

        manager = Manager()
        ShareQueue = manager.Queue()
        Locker = manager.Lock()

        p = Pool()
        for i in range(n_jobs):
            p.apply_async(taskFunction, args=(ShareQueue, Locker, partitionDataFunc(distributeList, i, n_jobs),
                                              parameterList, ))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')

        resultList = []
        while not ShareQueue.empty():
            resultList.append(ShareQueue.get_nowait())

        return reduce(reduceFunction, resultList)
    else:
        print('Parameter error')



# Example
def task(ShareQueue, Locker, data, parameterList):

    print('Child process %s.' % os.getpid())

    result = sum(data)

    Locker.acquire()
    ShareQueue.put(result)
    Locker.release()

if __name__ == "__main__":
    data = [e for e in range(100)]

    n_job = 4

    partitionFunc = lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i]

    print(multipleProcess(distributeList=data, partitionDataFunc=partitionFunc, taskFunction=task,
                          n_jobs=n_job, reduceFunction=lambda x,y:x+y, parameterList=[]))

