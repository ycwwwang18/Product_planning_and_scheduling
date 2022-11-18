# 主运行函数

import numpy as np
from GAv20221115 import *
from datav20221110 import *
from multiprocessing import Pool, cpu_count

global population
global ga


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


def dataPrepare(order_name, order_time, strategy, objective, population_size, crossover_rate, mutation_rate,
                select_rate, evolution_num, mutation_change_point):
    """生成遗传算法对象前的数据准备"""
    ga_data = getGa_data(order_name, order_time)
    global ga
    ga = GA(ga_data, strategy, objective, population_size, crossover_rate, mutation_rate, select_rate,
            evolution_num, mutation_change_point)


def gav20221109Execute():
    """执行遗传算法"""
    fitness_evolution = []  # 记录每一代的最优目标值
    best_chromosome = []  # 记录当前的最优个体
    best_objective_value = 0  # 记录当前的最优目标值
    execute_flag = True
    execute_count = 1

    #############初始化种群############
    global population
    while execute_flag:
        another_execute = False
        population = ga.initialPopulation(ga.population_size)
        #############进行若干次进化#############
        for i in range(ga.evolution_num):  # 进化次数
            print(
                f"----------------------------------------------第{execute_count}此执行GA，第{i + 1}代----------------------------------------------")
            ####################种群适应度计算####################
            decode_multiprocess = Multiprocess()
            decode_args = zip([ga] * len(population), population)
            decode_results = decode_multiprocess.work(ga.decodeChildTask, decode_args, i + 1)

            start_time_list = decode_results[0]
            end_time_list = decode_results[1]
            fitness_array = np.array(decode_results[2])
            schedule_list = decode_results[3]

            ####################获取种群最优个体####################
            best_index, best_objective_value, _, best_end, _ = ga.getBestChromosome(start_time_list, end_time_list,
                                                                                    fitness_array, schedule_list)

            best_chromosome = population[best_index]
            best_chromosome = np.array([best_chromosome])  # 用于最优个体的保存
            print("第%s代：最优个体的目标值为：%s，项目结束时间为：%s" % (i + 1, best_objective_value, best_end))
            fitness_evolution.append(best_objective_value)

            ###################手动控制进化的进程###################
            if i > 10:
                keyboard_input = input(
                    "按下c结束本次GA的运行，进行下一次GA运行；按下Enter键结束整体GA运行，输出最终结果；否则继续执行至本次GA：")
                if keyboard_input == 'c':  # 输入回车就跳出该循环,继续下一次外循环
                    another_execute = True
                    print(f"结束第{execute_count}次运行，进行下一次的运行。")
                    break
                elif keyboard_input == '':  # 输入Esc就跳出该循环,结束GA的运行，输出最终结果。
                    print(f"结束第{execute_count}次运行，输出最终结果：")
                    break

            ###################选择种群中的优质个体##################
            if i > 0:  # 初始的种群不用选择
                population = ga.select(population, fitness_array, ga.select_rate)

            ##################优质个体的交叉和变异###################
            crossover_population = ga.crossOver(population)
            mutation_population = ga.mutation(population)

            ######################得到下一代种群#####################
            offspring_population = np.vstack((crossover_population, mutation_population))
            population = np.vstack((offspring_population, best_chromosome))

        execute_flag = another_execute

        if not another_execute:
            """结果输出"""
            _, schedule_df, _ = ga.decodeChromosome(best_chromosome[0])
            project_end_time = schedule_df['End Time'].max()

            ga.resultExport(schedule_df, fitness_evolution, best_objective_value, project_end_time)
            print(
                "-----------------------------------------------------------------------------------------------------------")
            print("最好目标值为：" + str(best_objective_value))
            print("完工时间为：" + str(project_end_time))

        execute_count += 1


def GAv20221115Execute():
    """执行遗传算法"""
    fitness_evolution = []  # 记录每一代的最优目标值
    fitness_array = []  # 记录各代的适应度
    best_chromosome = []  # 记录当前的最优个体
    best_objective_value = 0  # 记录当前的最优目标值
    execute_flag = True
    execute_count = 1
    iterate_count_all = 1  # 记录总的迭代次数

    #############初始化种群############
    global population
    while execute_flag:
        another_execute = False
        iterate_flag = True
        iterate_count = 1

        ##############生成初始种群#############
        population = ga.initialPopulation(ga.population_size)

        #############进行若干次进化#############
        while iterate_flag:  # 进化次数
            print(
                f"----------------------------------------------第{execute_count}次执行GA，第{iterate_count}代----------------------------------------------")

            ##################选择种群中的优质个体#################
            if iterate_count > 1:  # 初始的种群不用选择
                population = ga.select(population, fitness_array, ga.select_rate, iterate_count)

            ######################交叉和变异#####################
            crossover_population = ga.crossOver(population)
            print("变异多进程开始进行：")
            mutation_num = int(ga.mutation_rate * ga.population_size)
            mutation_multiprocess = Multiprocess()
            mutation_args = zip([ga] * mutation_num, [population] * mutation_num, [iterate_count] * mutation_num)
            mutation_results = mutation_multiprocess.work(ga.mutationChildTask, mutation_args, iterate_count)
            mutation_population = np.array(mutation_results[0])

            ######################得到下一代种群##################
            offspring_population = np.vstack((crossover_population, mutation_population))
            if isinstance(best_chromosome, np.ndarray):
                population = np.vstack((offspring_population, best_chromosome))
            else: population = offspring_population
            # population = np.vstack((population, offspring_population))

            ####################种群适应度计算####################
            print("解码多进程开始进行：")
            decode_multiprocess = Multiprocess()
            decode_args = zip([ga] * len(population), population)
            decode_results = decode_multiprocess.work(ga.decodeChildTask, decode_args, iterate_count)
            fitness_array = np.array(decode_results[2])  # 把适应度list转化为array

            ####################保存种群最优个体###################
            best_index, best_objective_value, _, best_end, _ = ga.getBestChromosome(decode_results[0],
                                                                                    decode_results[1],
                                                                                    fitness_array, decode_results[3])
            best_chromosome = population[best_index]
            # 记录本代的最优目标值
            fitness_evolution.append(best_objective_value)
            best_chromosome = np.array([best_chromosome])  # 用于最优个体的保存到下一代
            print("第%s代：最优个体的目标值为：%s，项目结束时间为：%s" % (iterate_count, best_objective_value, best_end))

            ########################结束判断#######################
            def isStop():
                nonlocal iterate_flag, another_execute

                if iterate_count_all > ga.evolution_num:
                    print(f"迭代进化次数超过{ga.evolution_num}，结束GA运行，输出最终结果。")
                    iterate_flag = False
                    another_execute = False
                elif iterate_count_all > 6:
                    best_obj_1 = fitness_evolution[iterate_count_all - 2]
                    best_obj_2 = fitness_evolution[iterate_count_all - 3]
                    best_obj_3 = fitness_evolution[iterate_count_all - 4]
                    best_obj_4 = fitness_evolution[iterate_count_all - 5]
                    best_obj_5 = fitness_evolution[iterate_count_all - 6]
                    # 计算此代的最优值与前5代最优值之间的平方差
                    MSE = pow(best_objective_value - best_obj_1, 2) + \
                          pow(best_objective_value - best_obj_2, 2) + \
                          pow(best_objective_value - best_obj_3, 2) + \
                          pow(best_objective_value - best_obj_4, 2) + \
                          pow(best_objective_value - best_obj_5, 2)
                    if MSE < 0.05:  # 最优值6代内不再变化
                        print("最优值6代内不再变化，终止本次GA运行，开启一次新的GA运行。")
                        iterate_flag = False
                        another_execute = True

                # 通过外界输入来控制进程
                if iterate_count_all > 25:
                    key_board_input = input("按Enter结束GA运行，输出最终结果；按c终止本次GA运行，开启新一次的GA运行；否则继续运行：")
                    if key_board_input == '':
                        print("结束GA运行，输出最终结果。")
                        iterate_flag = False
                        another_execute = False
                    elif key_board_input == 'c':
                        print(f"终止第{execute_count}次GA运行，开启下一次GA运行。")
                        iterate_flag = False
                        another_execute = True
                    else: print(f"继续本次的GA运行，进入第{iterate_count+1}次迭代进化。")

            iterate_count += 1
            iterate_count_all += 1
            isStop()

        execute_flag = another_execute

        if not another_execute:
            """结果输出"""
            _, schedule_df, _ = ga.decodeChromosome(best_chromosome[0])
            project_end_time = schedule_df['End Time'].max()

            ga.resultExport(schedule_df, fitness_evolution, best_objective_value, project_end_time)
            print(
                "-----------------------------------------------------------------------------------------------------------")
            print("最好目标值为：" + str(best_objective_value))
            print("完工时间为：" + str(project_end_time))

        execute_count += 1


def run():
    """遗传算法"""
    order_name = "order1"
    order_time = "2021-12-31"
    strategy = 1
    algorithm = "GA"

    if algorithm == "GA":
        objective = 2
        population_size = 10
        crossover_rate = 0.8
        mutation_rate = 0.1
        select_rate = 0.8
        mutation_change_point = 30
        evolution_num = 1
        dataPrepare(order_name, order_time, strategy, objective, population_size, crossover_rate,
                    mutation_rate, select_rate, evolution_num, mutation_change_point)
        GAv20221115Execute()


if __name__ == '__main__':
    run()
