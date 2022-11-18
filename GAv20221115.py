import os
import json
import sys

from datav20221110 import *
from pylab import *
from copy import deepcopy
import plotly.offline
import plotly.express as px
import matplotlib.pyplot as plt
from itertools import permutations

mpl.rcParams['font.sans-serif'] = ['SimHei']

"""
更新记录：
1、选择：select_rate的自适应。随着进化代数的增加，原种群的比例逐渐减少，增加随机生成解的比例，提高解的多样性，跳出局部最优。
2、变异算子：降低R的上界，增加片段内的变异次数，提高局部搜索能力。设计两种变异算子，随着迭代的进行，从全排列算子转变为随机排列算子。
3、增加了结束的判断函数。
"""


class GA:
    """遗传算法类"""

    def __init__(self, data, strategy, objective, population_size, crossover_rate, mutation_rate, select_rate,
                 evolution_num, mutation_change_point):
        """
        :param data: 传入遗传算法的对应数据类GAData
        :param strategy: 启发式策略，编号
        :param objective: 优化目标
        :param population_size: 种群规模
        :param crossover_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param select_rate: 种群选择的比例
        :param evolution_num: 进化次数
        """
        abs_path = os.path.abspath(__file__)
        _, filename = os.path.split(abs_path)
        filename = filename[:-3]
        self.file_name = filename + f"Strategy{strategy}Obj{objective}"

        self.population_size = population_size  # 种群规模
        self.crossover_rate = crossover_rate  # 交叉概率
        self.mutation_rate = mutation_rate  # 变异概率
        self.select_rate = select_rate  # 选择比例
        self.mutation_change_point = mutation_change_point
        self.evolution_num = evolution_num  # 进化次数
        self.objective = objective  # 优化目标
        self.cal = CalculateUtils()

        if strategy == 1:  # 使用“同一产品类别一起加工”的策略
            self.data = data
            self.job_id_list = data.category_list  # 任务编号，list
            self.jobs = data.jobs  # 任务集合
            self.job_num = data.job_num  # 任务总数
            self.machines = data.machines  # 机器集合
            self.machine_num = data.machine_num  # 机器总数
            self.order_batches = data.order_batches  # 批量订单集合
            self.chromosome_size = data.chromosome_size  # 染色体长度
            self.procedure_num = data.procedure_num  # 各任务的工序数量，list

    '''种群初始化'''

    def initialPopulation(self, chromosome_num):
        """编码，返回初始解array"""
        # TODO：以下程序是不是通用于各种策略?
        population = np.empty(shape=(chromosome_num, self.chromosome_size, 2)).astype(int)  # 2表示一个是工序，一个是对应的机器

        for i in range(chromosome_num):  # 遍历种群中的所有染色体

            # 染色体任务部分的数据生成
            job_data = np.array(self.job_id_list)
            job_data = np.repeat(job_data, self.procedure_num)
            np.random.shuffle(job_data)

            # 染色体构造
            for j in range(self.chromosome_size):  # 遍历染色体上的所有基因，遍历job_data

                # 将任务部分的数据指定到染色体的相应基因位置
                job_num = job_data[j]  # 任务编号
                population[i][j][0] = job_num

                # 随机选择各个任务的一个加工机器，并指定到染色体的相应基因位置
                procedure_show_times = np.sum(job_data[0:j + 1] == job_num)  # 某任务编号是第几次出现，从而得出这是该任务的第几道工序
                population[i][j][1] = choice(
                    self.jobs.loc[job_num].machines[procedure_show_times - 1])  # 从该任务的工序对应的可选机器列表中随机选择一个

        return population

    '''解码：①调度表，②目标值'''

    def decodeJob(self, is_first_job_of_machine, machine_id, job_id, result):
        """对某机器上的任务进行解码"""
        # TODO：修改解码方式

        # 数据准备
        '''结果存储，生成dataframe'''
        start_time_list = []
        end_time_list = []
        machine_list = []
        machine_status_list = []
        category_id_list = []
        order_id_list = []
        product_id_list = []
        num_list = []

        this_job = self.jobs.loc[job_id]
        this_machine = self.machines.loc[machine_id]
        order_batch = self.order_batches.loc[job_id]
        maintenance_day0 = this_machine.maintenance_day[0]  # 机器当月的维修日期
        maintenance_day1 = this_machine.maintenance_day[1]  # 机器下月的维修日期

        def linesAppend(machine, status, start_time, end_time, category_ID, order_ID, product_ID, num_index):
            """往结果中增添一行"""
            start_time_list.append(start_time)
            end_time_list.append(end_time)
            machine_list.append(machine)
            machine_status_list.append(status)
            category_id_list.append(category_ID)
            order_id_list.append(order_ID)
            product_id_list.append(product_ID)
            num_list.append(num_index)

        model_index = 0
        for model_id, order_by_model in order_batch.items():  # 遍历该类别下的每个型号
            product_id = model_id
            for order_index, order in enumerate(order_by_model):  # 遍历型号对应的每个订单
                order_id = order.order_id
                item_num = order.product_number
                for item_index in range(item_num):  # 遍历一个订单中的每一件产品
                    num = item_index + 1  # 产品序号

                    flag_maintenance = 0  # 设备维修保养状态标记
                    flag_idle = 0  # 设备空转状态标记
                    flag_shutdown = 0  # 设备停机状态标记
                    duration_start_time = 0  # 设备生产间隔的开始时间
                    duration_end_time = 0  # 设备生产间隔的结束时间
                    maintenance_start_time = 0  # 设备维修开始时间
                    maintenance_end_time = 0  # 设备维修结束时间

                    # 计算产品的加工开始时间
                    if (is_first_job_of_machine == 1) & (model_index == 0) & (order_index == 0) & (
                            item_index == 0):  # 机器上的第一个job的第一个产品
                        item_start_time = this_job.latest_end_time  # 该job的上一道工序结束时间
                    elif (model_index == 0) & (order_index == 0) & (item_index == 0):  # 机器上的某个job的第一个产品（更换job时），需要换型
                        change_start_time = this_machine.latest_end_time  # 换型开始时间
                        change_end_time = change_start_time + this_machine.model_change_time
                        item_start_time = max(change_end_time,
                                              this_job.latest_end_time)  # 该job的上一道工序结束时间和机器上一次加工结束时间之间的最大值
                        # 判断两个job的生产间隔中机器的状态
                        if this_machine.is_shutdown_when_change_model:  # 如果换型需要停机，则机器停机
                            flag_shutdown = 1
                        elif item_start_time - this_machine.latest_end_time > this_machine.max_idle_time:
                            # 如果换型不需停机，则判断间隔时间是否超过机器最大空转时间，如果超过，则机器停机；否则机器空转
                            flag_shutdown = 1
                        else:
                            flag_idle = 1
                        duration_start_time = this_machine.latest_end_time  # 两个job间隔时间段的起点
                        duration_end_time = item_start_time  # 两个job间隔时间段的终点
                    elif (order_index == 0) & (item_index == 0):  # 如果是某一型号的第一个产品，生产该产品前需要加上换型时间
                        if this_machine.is_shutdown_when_change_model:
                            flag_shutdown = 1
                        else:
                            flag_idle = 1
                        duration_start_time = this_machine.latest_end_time  # 换型的开始时间
                        duration_end_time = duration_start_time + this_machine.model_change_time  # 换型的结束时间
                        item_start_time = duration_end_time
                    else:
                        item_start_time = this_machine.latest_end_time

                    # 判断产品的加工开始时间是否在维保期
                    if maintenance_day0 <= item_start_time <= maintenance_day0 + np.timedelta64(24, 'h'):
                        flag_maintenance = 1
                        maintenance_start_time = maintenance_day0
                        maintenance_end_time = maintenance_day0 + np.timedelta64(24, 'h')
                        item_start_time = maintenance_end_time  # 在维保期后才能开始生产，TODO：尝试其他的策略
                    if maintenance_day1 <= item_start_time <= maintenance_day1 + np.timedelta64(24, 'h'):
                        flag_maintenance = 1
                        maintenance_start_time = maintenance_day1
                        maintenance_end_time = maintenance_day1 + np.timedelta64(24, 'h')
                        item_start_time = maintenance_end_time  # 在维保期后才能开始生产，TODO：尝试其他的策略

                    if (is_first_job_of_machine == 1) & (model_index == 0) & (order_index == 0) & (item_index == 0):
                        this_machine.first_start_time = item_start_time  # 该机器第一次的开机时间

                    # 计算产品的加工结束时间
                    item_end_time = item_start_time + this_machine.processing_time.loc[job_id]

                    # 保存机器空转的时间段
                    if flag_idle == 1:
                        job_segment_idle = {
                            "start": self.cal.time_to_str(duration_start_time),
                            "end": self.cal.time_to_str(duration_end_time),
                            "val": [10]
                        }
                        result[machine_id - 1]["data"].append(job_segment_idle)
                        linesAppend(machine_id, '空转', duration_start_time, duration_end_time, '空转', '', '', '')

                    # 保存机器停机维修保养的时间段
                    if flag_maintenance == 1:
                        job_segment_maintenance = {
                            "start": self.cal.time_to_str(maintenance_start_time),
                            "end": self.cal.time_to_str(maintenance_end_time),
                            "val": [1]
                        }
                        result[machine_id - 1]["data"].append(job_segment_maintenance)
                        linesAppend(machine_id, '维保', maintenance_start_time, maintenance_end_time, '维保', '', '',
                                    '')

                    # 保存机器停机的时间段
                    if flag_shutdown == 1:
                        job_segment_shutdown = {
                            "start": self.cal.time_to_str(duration_start_time),
                            "end": self.cal.time_to_str(duration_end_time),
                            "val": [0]
                        }
                        result[machine_id - 1]["data"].append(job_segment_shutdown)
                        linesAppend(machine_id, '停机', duration_start_time, duration_end_time, '停机', '', '', '')

                    # 保存机器生产的任务段
                    job_segment = {
                        "start": self.cal.time_to_str(item_start_time),
                        "end": self.cal.time_to_str(item_end_time),
                        "val": [20, order_id, product_id, item_num]
                    }
                    result[machine_id - 1]["data"].append(job_segment)
                    linesAppend(machine_id, '生产', item_start_time, item_end_time, job_id, order_id, product_id, num)

                    this_machine.latest_end_time = item_end_time
                    this_job.latest_end_time = item_end_time
            model_index += 1

        result_df = pd.DataFrame(list(
            zip(machine_list, start_time_list, end_time_list, machine_status_list, category_id_list, order_id_list,
                product_id_list,
                num_list)))
        result_df.columns = ['Machine', 'Start Time', 'End Time', 'Machine Status', 'Category ID', 'Order ID',
                             'Product ID', '#']
        return result, result_df

    def decodeChromosome(self, chromosome):
        """对染色体进行解码，返回json格式和dataframe格式的调度表，以及机器第一次开机时间"""

        # 数据准备
        result_schedule = []  # 调度表
        schedule_df = pd.DataFrame(
            columns=['Machine', 'Start Time', 'End Time', 'Machine Status', 'Category ID', 'Order ID', 'Product ID',
                     '#'])  # dataframe格式的结果
        for i in range(self.machine_num):
            shop = {
                "name": "Shop" + str(i + 1),
                "data": []
            }
            result_schedule.append(shop)
        job_count_of_machine = [0] * self.machine_num  # 用于计算某机器上job的累计个数
        machine_first_start_time = []  # 机器第一次开机时间

        for i in range(self.chromosome_size):  # 遍历染色体上的所有基因
            job_id = chromosome[i][0]  # 任务编号
            machine_id = chromosome[i][1]  # 机器编号
            job_count_of_machine[machine_id - 1] += 1
            if job_count_of_machine[machine_id - 1] == 1:  # 该job是这个机器上的第一个加工任务
                result_schedule, result_df = \
                    self.decodeJob(1, machine_id, job_id, result_schedule)
            else:
                result_schedule, result_df = \
                    self.decodeJob(0, machine_id, job_id, result_schedule)
            schedule_df = pd.concat([schedule_df, result_df])

        for machine in self.machines:
            machine_first_start_time.append(machine.first_start_time)

        self.data.resetMachine(self.machines)
        self.data.resetJob(self.jobs)
        schedule_df.index = [_ for _ in range(schedule_df.shape[0])]
        return result_schedule, schedule_df, machine_first_start_time

    def getObjectiveValue(self, chromosome):
        """获取个体的目标值"""
        if self.objective == 2:
            _, schedule_dataframe, machine_first_start_time = self.decodeChromosome(chromosome)
            project_start_time = schedule_dataframe['Start Time'].min()
            project_end_time = schedule_dataframe['End Time'].max()
            objective_value = self.cal.getEnergyCost(schedule_dataframe, machine_first_start_time)
            return project_start_time, project_end_time, objective_value, schedule_dataframe

    def getFitness(self, query_population):
        """获取种群的适应度，返回适应度的array"""
        fitness_list = []
        project_start_time_list = []
        project_end_time_list = []
        schedule_list = []
        if self.objective == 2:  # 能耗成本最小化
            for i in range(len(query_population)):  # 遍历种群中的每个个体
                # 对于一个新的个体，就要重置jobs和machines（latest end time，first start time），实例对象属性绑定
                project_start_time, project_end_time, objective_value, schedule_dataframe = self.getObjectiveValue(
                    query_population[i])
                fitness = 1 / objective_value  # TODO：适应度的计算方式可以替换
                fitness_list.append(fitness)
                project_start_time_list.append(project_start_time)
                project_end_time_list.append(project_end_time)
                schedule_list.append(schedule_dataframe)
        fitness_array = np.array(fitness_list)
        return project_start_time_list, project_end_time_list, fitness_array, schedule_list

    @staticmethod
    def getBestChromosome(project_start_time_list, project_end_time_list, fitness_array, schedule_list):
        """获取种群中的最好个体，返回最优个体和最优目标值"""
        best_fitness = np.max(fitness_array)
        best_index = np.argmax(fitness_array)
        best_objective_value = 1 / best_fitness
        best_start_time = project_start_time_list[best_index]
        best_end_time = project_end_time_list[best_index]
        best_schedule = schedule_list[best_index]

        return best_index, best_objective_value, best_start_time, best_end_time, best_schedule

    '''约束条件检查'''

    def chromosomeCheck(self, chromosome):
        """检查个体是否符合约束条件"""
        is_feasible = True  # 满足约束为真，不满足为假
        process_num = np.zeros(shape=self.job_num).astype(int) - 1  # 存储当前工序是对应任务的第几道工序 -1
        for j in range(self.chromosome_size):  # 对染色体进行遍历
            job_id = chromosome[j][0]
            machine_id = chromosome[j][1]
            process_num[job_id - 1] += 1
            process_index = process_num[job_id - 1]
            if machine_id not in self.jobs.loc[job_id].machines[process_index]:
                is_feasible = False
                break
        return is_feasible

    def chromosomeCorrecting(self, chromosome):
        """将不符合约束条件的染色体进行校正"""
        process_num = np.zeros(shape=self.job_num).astype(int) - 1  # 存储当前工序是对应任务的第几道工序 -1

        for j in range(self.chromosome_size):  # 对染色体进行遍历
            job_id = chromosome[j][0]
            machine_id = chromosome[j][1]
            process_num[job_id - 1] += 1
            process_index = process_num[job_id - 1]
            if machine_id not in self.jobs.loc[job_id].machines[process_index]:
                chromosome[j][1] = choice(self.jobs.loc[job_id].machines[process_index])

        return chromosome

    def populationCorrecting(self, population):
        """对整个种群进行校正"""
        for i in range(len(population)):
            population[i] = self.chromosomeCorrecting(population[i])
        return population

    def populationFilter(self, population):
        """过滤种群中不符合约束条件的个体"""
        new_population = np.array(list(filter(self.chromosomeCheck, population)))
        return new_population

    '''染色体交叉'''

    def crossOperator(self, population):
        """个体交叉算子"""
        random_parent_index = np.random.randint(0, len(population), 2)  # 随机抽取两个个体
        parent0 = deepcopy(population[random_parent_index[0]]).tolist()
        parent1 = deepcopy(population[random_parent_index[1]]).tolist()
        random_job = np.random.randint(1, 1 + self.job_num)  # 随机选择一个job
        child0 = np.empty(shape=(self.chromosome_size, 2)).astype(int)
        child1 = np.empty(shape=(self.chromosome_size, 2)).astype(int)
        # 交叉生成第一个子代
        for i in range(self.chromosome_size):  # 对染色体进行遍历
            if parent0[i][0] == random_job:
                child0[i] = parent0[i]
            else:
                x = parent1.pop(0)
                while x[0] == random_job:
                    x = parent1.pop(0)
                child0[i] = x
        # 交叉生成第二个子代
        parent0 = deepcopy(population[random_parent_index[0]]).tolist()
        parent1 = deepcopy(population[random_parent_index[1]]).tolist()
        for i in range(self.chromosome_size):
            if parent1[i][0] == random_job:
                child1[i] = parent1[i]
            else:
                y = parent0.pop(0)
                while y[0] == random_job:
                    y = parent0.pop(0)
                child1[i] = y
        return child0, child1

    def crossOver(self, population):
        """种群交叉"""

        cross_num = int(self.crossover_rate * self.population_size)
        offspring_population = np.empty(shape=(2 * cross_num, self.chromosome_size, 2)).astype(int)
        for i in range(cross_num):
            results = self.crossOperator(population)
            offspring_population[2 * i] = results[0]
            offspring_population[2 * i + 1] = results[1]
        offspring_population = self.populationFilter(offspring_population)
        return offspring_population

    '''染色体变异'''

    def mutationOperator(self, population, iterate_num):
        """变异算子"""
        random_parent_index = np.random.randint(0, len(population))  # 随机抽取一个个体
        R = np.random.randint(1, 4)  # 修改了这个, 不超过3
        random_point = np.random.randint(0, self.chromosome_size - R + 1)  # 随机选取一个变异片段的端点
        offspring = deepcopy(population[random_parent_index])
        if iterate_num > self.mutation_change_point:
            """对随机选取的片段进行打乱顺序"""
            np.random.shuffle(offspring[random_point:random_point + R])
            offspring = np.array([offspring])
            offspring = self.populationCorrecting(offspring)
            return offspring
        else:
            """对随机选取的片段进行全排列"""
            parent = deepcopy(population[random_parent_index])
            offsprings = np.empty(shape=(math.factorial(R), self.chromosome_size, 2)).astype(int)  # 用于存储变异得到的所有子代
            for index, element in enumerate(permutations(parent[random_point:random_point + R])):
                offspring[random_point:random_point + R] = list(element)
                offsprings[index] = offspring
            offsprings = self.populationCorrecting(offsprings)
            project_start_time_list, project_end_time_list, fitness_array, schedule_list = self.getFitness(offsprings)
            best_index, _, _, _, _ = self.getBestChromosome(project_start_time_list, project_end_time_list,
                                                            fitness_array, schedule_list)
            best_offspring = offsprings[best_index]
            best_offspring = np.array([best_offspring])
            return best_offspring

    def mutation(self, population, iterate_num):
        """种群变异"""
        mutation_num = int(self.mutation_rate * self.population_size)
        offspring_population = []
        for i in range(mutation_num):
            if i == 0:
                offspring_population = self.mutationOperator(population, iterate_num)
                while offspring_population.size == 0:  # 如果没有变异出满足约束的个体，则重复操作，直至有满足约束和个体出现
                    offspring_population = self.mutationOperator(population, iterate_num)
            else:
                mutation_result = self.mutationOperator(population, iterate_num)
                while mutation_result.size == 0:
                    mutation_result = self.mutationOperator(population, iterate_num)
                offspring_population = np.append(offspring_population, mutation_result, axis=0)
        return offspring_population

    '''选择'''

    def select(self, population, fitness_array, select_rate, iterate_num):
        select_rate = select_rate - iterate_num / self.population_size / 2
        sort_index = np.argsort(-fitness_array)
        population_size = len(population)
        select_num = int(population_size * select_rate)
        select_population = population[sort_index][:select_num]
        new_population = self.initialPopulation(population_size - select_num)
        return np.vstack((select_population, new_population))

    '''结果输出'''

    def resultExport(self, schedule, fitness, obj_val, end):
        obj_val = int(obj_val)
        end = end.strftime('%Y%m%d%H%M')
        file_name = self.file_name + f"ObjVal{obj_val}End{end}"

        def exportSchedule():
            """调度表"""
            schedule_path = "调度表" + file_name + ".xlsx"
            schedule.to_excel(schedule_path)

        def plotFitnessEvolution():
            """目标值迭代曲线"""
            evolution_name = "进化" + file_name
            y_value = fitness
            len_y = len(y_value)
            x_value = [i for i in range(1, len_y + 1)]
            plt.plot(x_value, y_value)
            x_major_locator = MultipleLocator(10)
            ax = plt.gca()  # 获取轴
            ax.xaxis.set_major_locator(x_major_locator)
            plt.xticks(x_value, x_value)
            plt.margins(0)
            plt.xlabel(u"迭代次数")
            plt.ylabel(u"目标值")
            plt.title("目标值迭代曲线")
            # 先保存再show，否则保存的图片可能是空白的
            plt.savefig(fname=evolution_name)  # 算法、策略
            plt.show()

        def plotGantt():
            """甘特图"""
            gantt_png_path = "甘特" + file_name + ".png"
            gantt_html_path = "甘特" + file_name + ".html"

            # 对schedule进行调整
            schedule_reformat = schedule[
                (schedule['Machine Status'] == '生产') | (schedule['Machine Status'] == '空转')]
            # schedule_reformat.loc[:, 'Machine'] = schedule_reformat.Machine.map(lambda machineid: "M" + str(machineid))
            # schedule_reformat = schedule_reformat.sort_values(by='Machine', axis=0, ascending=True, inplace=False)
            machine_list = schedule_reformat.Machine.unique()
            machine_list = np.sort(machine_list)
            machine_name_list = list(map(lambda x: 'M'+str(x), machine_list))
            schedule_reformat.loc[:, 'Machine'] = schedule_reformat.Machine.map(lambda machineid: "M" + str(machineid))
            fig = px.timeline(schedule_reformat, x_start='Start Time', x_end='End Time', y='Machine',
                              color='Category ID',
                              color_discrete_map={1: 'red', 2: 'green', 3: 'blue', 4: 'goldenrod', 5: 'magenta',
                                                  6: 'purple', 7: 'yellow', 8: 'pink', '空转': 'grey'},
                              hover_name='Order ID',
                              category_orders={'Machine': machine_name_list,
                                               'Category ID': [1, 2, 3, 4, 5, 6, 7, 8, '空转']})
            fig.update_yaxes(showgrid=True, griddash='dash')
            fig.write_image(gantt_png_path, width=7000, height=4000)
            plotly.offline.plot(fig, filename=gantt_html_path)
            fig.show()

        exportSchedule()
        plotFitnessEvolution()
        plotGantt()

    @staticmethod
    def decodeChildTask(ga_data, chromosome):
        """用于多进程解码的实现，个体的解码函数"""
        print("\033[1;32m", "█", "\033[0m", sep="", end="")
        if ga_data.objective == 2:
            _, schedule_dataframe, machine_first_start_time = ga_data.decodeChromosome(chromosome)
            project_start_time = schedule_dataframe['Start Time'].min()
            project_end_time = schedule_dataframe['End Time'].max()
            objective_value = ga_data.cal.getEnergyCost(schedule_dataframe, machine_first_start_time)
            return project_start_time, project_end_time, 1 / objective_value, schedule_dataframe
        time.sleep(0.01)

    @staticmethod
    def mutationChildTask(ga_data, population, iterate_num):
        """用于多进程变异的实现，种群变异函数"""
        print("\033[1;32m", "█", "\033[0m", sep="", end="")
        random_parent_index = np.random.randint(0, len(population))  # 随机抽取一个个体
        R = np.random.randint(1, 4)  # 修改了这个, 不超过3
        random_point = np.random.randint(0, ga_data.chromosome_size - R + 1)  # 随机选取一个变异片段的端点
        offspring = deepcopy(population[random_parent_index])
        if iterate_num > ga_data.mutation_change_point:
            """对随机选取的片段进行打乱顺序"""
            np.random.shuffle(offspring[random_point:random_point + R])
            offspring = np.array([offspring])
            offspring = ga_data.populationCorrecting(offspring)
            return offspring
        else:
            """对随机选取的片段进行全排列"""
            parent = deepcopy(population[random_parent_index])
            offsprings = np.empty(shape=(math.factorial(R), ga_data.chromosome_size, 2)).astype(int)  # 用于存储变异得到的所有子代
            for index, element in enumerate(permutations(parent[random_point:random_point + R])):
                offspring[random_point:random_point + R] = list(element)
                offsprings[index] = offspring
            offsprings = ga_data.populationCorrecting(offsprings)
            project_start_time_list, project_end_time_list, fitness_array, schedule_list = ga_data.getFitness(
                offsprings)
            best_index, _, _, _, _ = ga_data.getBestChromosome(project_start_time_list, project_end_time_list,
                                                               fitness_array, schedule_list)
            best_offspring = offsprings[best_index]
            best_offspring = np.array([best_offspring])
            return best_offspring
