import os

from UCTB.utils import multiple_process


def task_func(share_queue, locker, data, parameters):

    print('Child process %s with pid %s' % (parameters[0], os.getpid()))

    for task in data:
        print('Child process', parameters[0], 'running', task)
        exec_str = 'python ARIMA.py --Dataset %s --City %s ' % (task[0], task[1])
        if task[2] != '':
            exec_str += task[2]
        os.system(exec_str)

    locker.acquire()
    share_queue.put(None)
    locker.release()


if __name__ == '__main__':

    task_list = [
        ['Bike', 'NYC', ''],
        ['Bike', 'Chicago', ''],
        ['Bike', 'DC', ''],
        ['Metro', 'Chongqing', ''],
        ['Metro', 'Shanghai', ''],
        ['DiDi', 'Chengdu', ''],
        ['DiDi', 'Xian', ''],
        ['ChargeStation', 'Beijing', '']
    ]

    n_jobs = 1

    multiple_process(distribute_list=task_list,
                     partition_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
                     task_func=task_func, n_jobs=n_jobs,
                     reduce_func=lambda x,y: None, parameters=[])

