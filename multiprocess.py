import os
import time
from multiprocessing import Pool, cpu_count


class Multiprocess:
    """用于实现多进程"""

    def __init__(self):
        self.pool = Pool(processes=cpu_count())

    def work(self, func, args, iterate_num):
        """执行多进程的一项工作，须输入函数和可迭代的参数集合，以及此项工作属于第几次迭代"""
        results = []
        print(f"第{iterate_num}个父进程开始：", end=" |")
        work_start_time = time.time()

        for arg in args:
            results.append(self.pool.apply_async(func, arg))
        self.pool.close()
        self.pool.join()
        work_end_time = time.time()
        work_duration = (work_end_time - work_start_time) / 60
        print("|  ", "\033[1;32m", f"{len(results)}", "\033[0m", sep="")
        print(f"父进程{os.getpid()}结束，耗时{work_duration}分钟。")
        result_list = self.getResults(results)
        return result_list

    @staticmethod
    def getResults(results):
        """将结果存储为list"""
        result_column_num = len(results[0].get())  # 函数返回的变量个数
        result_list = [[] for _ in range(result_column_num)]
        for result in results:
            result = result.get()
            for i, result_col in enumerate(result):
                result_list[i].append(result_col)
        return result_list
