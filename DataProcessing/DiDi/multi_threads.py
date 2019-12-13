import os

from multiprocessing import Pool, Manager
from functools import reduce


# (my_rank, n_jobs, dataList, resultHandleFunction, parameterList)
def multiple_process(distribute_list, partition_func, task_func, n_jobs, reduce_func, parameters):
    if callable(partition_func) and callable(task_func) and callable(reduce_func):
        print('Parent process %s.' % os.getpid())

        manager = Manager()
        share_queue = manager.Queue()
        locker = manager.Lock()

        p = Pool()
        for i in range(n_jobs):
            p.apply_async(task_func, args=(share_queue, locker, partition_func(distribute_list, i, n_jobs),
                                           [i] + parameters,))
        print('Waiting for all sub_processes done...')
        p.close()
        p.join()
        print('All sub_processes done.')

        result_list = []
        while not share_queue.empty():
            result_list.append(share_queue.get_nowait())

        return reduce(reduce_func, result_list)
    else:
        print('Parameter error')


# Example
def task(share_queue, locker, data, parameters):

    print('Child process %s with pid %s' % (parameters[0], os.getpid()))

    result = sum(data)

    locker.acquire()
    share_queue.put(result)
    locker.release()


if __name__ == "__main__":

    data = [e for e in range(1000000)]

    n_job = 4

    sum_result = \
        multiple_process(distribute_list=data,
                         partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
                         task_func=task, n_jobs=n_job, reduce_func=lambda x, y: x + y, parameters=[])

    print('Result', sum_result)


