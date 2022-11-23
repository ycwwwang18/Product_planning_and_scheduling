# 主运行函数
from GAv20221122 import *
from datav20221110 import *

global population
global ga


def dataPrepare(order_name, order_time, strategy, objective, population_size, crossover_rate, mutation_rate,
                select_rate, best_keep_num, evolution_num, mutation_change_point):
    """生成遗传算法对象前的数据准备"""
    ga_data = getGa_data(order_name, order_time)
    global ga
    ga = GA(ga_data, strategy, objective, population_size, crossover_rate, mutation_rate, select_rate, best_keep_num,
            evolution_num, mutation_change_point)


def run():
    """遗传算法"""
    order_name = "order1"
    order_time = "2021-12-31"
    strategy = 1
    algorithm = "GA"

    if algorithm == "GA":
        objective = 2
        population_size = 100
        crossover_rate = 0.8
        mutation_rate = 0.1
        select_rate = 0.8
        best_keep_num = 16
        mutation_change_point = 30
        evolution_num = 100
        dataPrepare(order_name, order_time, strategy, objective, population_size, crossover_rate,
                    mutation_rate, select_rate, best_keep_num, evolution_num, mutation_change_point)
        ga.execute()


if __name__ == '__main__':
    run()
