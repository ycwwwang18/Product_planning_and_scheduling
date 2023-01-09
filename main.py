# 主运行函数
from GAv20221207 import *
from data import *


def run():
    """遗传算法"""
    # 设置数据
    file_name = '题目2.dataset-v2.xlsx'
    order_name = 'order1'
    order_time = '2021-12-31'
    # 设置管理策略
    job_strategy = 'category_together'
    machine_strategy = 'machine_selection'
    time_strategy = 'no_work_at_night'
    nwn_rate = 0.3  # 不能在晚上生产的设备比例
    wn_rate = 0.3  # 只能在晚上生产的设备比例
    strategy_data = getStrategyData(file_name, order_name, order_time, job_strategy, machine_strategy, time_strategy, nwn_rate, wn_rate)
    algorithm = 'GA'

    if algorithm == 'GA':
        # 设置遗传算法的参数
        population_size = 100
        crossover_rate = 0.7
        mutation_rate = 0.1
        select_rate = 0.8
        best_keep_rate = 0.5
        mutation_change_point = 30
        evolution_num = 100

        ga_data = GAData(strategy_data)
        strategy = job_strategy + machine_strategy + time_strategy
        ga = GA(ga_data, strategy, population_size, crossover_rate, mutation_rate, select_rate, best_keep_rate,
                evolution_num, mutation_change_point)
        ga.execute()


if __name__ == '__main__':
    run()
