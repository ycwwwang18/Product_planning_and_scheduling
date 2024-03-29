from data import *
from multiprocess import *
from pylab import *
from copy import deepcopy
import plotly.offline
import plotly.express as px
import matplotlib.pyplot as plt
from itertools import permutations

mpl.rcParams['font.sans-serif'] = ['SimHei']


class GA:
    """遗传算法类"""

    def __init__(self, data, strategy, population_size, crossover_rate, mutation_rate, select_rate,
                 best_keep_rate,
                 evolution_num, mutation_change_point):
        """
        :param data: 传入遗传算法的对应数据类GAData
        :param strategy: 启发式策略，编号
        :param population_size: 种群规模
        :param crossover_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param select_rate: 种群选择的比例
        :param best_keep_rate: 保留最优个体的比例
        :param evolution_num: 进化次数
        :param mutation_change_point: 变异算子改变的时点
        """

        self.population_size = population_size  # 种群规模
        self.crossover_rate = crossover_rate  # 交叉概率
        self.mutation_rate = mutation_rate  # 变异概率
        self.select_rate = select_rate  # 选择比例
        self.best_keep_rate = best_keep_rate  # 最优保留的数量
        self.mutation_change_point = mutation_change_point
        self.evolution_num = evolution_num  # 进化次数
        self.cal = CalculateUtils()

        self.data = data
        self.job_id_list = data.job_id_list  # 任务编号，list
        self.jobs = data.jobs  # 任务集合
        self.job_num = data.job_num  # 任务总数
        self.machine_id_list = data.machine_id_list
        self.machines = data.machines  # 机器集合
        self.machine_num = data.machine_num  # 机器总数
        self.chromosome_size = data.chromosome_size  # 染色体长度

        abs_path = os.path.abspath(__file__)
        _, filename = os.path.split(abs_path)
        filename = filename[:-3]
        self.file_name = filename + f"Strategy{strategy}PopulationSize{population_size}CR{crossover_rate}MR{mutation_rate}SR{select_rate}BR{best_keep_rate}MCP{mutation_change_point}EvolutionNum{evolution_num}"

    '''种群初始化'''

    def initialPopulation(self, chromosome_num):
        """编码，返回初始解array"""
        population = np.empty(shape=(chromosome_num, self.chromosome_size, 2)).astype(int)  # 2表示一个是工序，一个是对应的机器

        for i in range(chromosome_num):  # 遍历种群中的所有染色体

            # 染色体任务部分的数据生成
            job_data = np.array(self.job_id_list)
            job_data = np.repeat(job_data, list(map(lambda x: x.procedure_num, self.jobs)))
            np.random.shuffle(job_data)

            # 染色体构造
            for j in range(self.chromosome_size):  # 遍历染色体上的所有基因，遍历job_data

                # 将任务部分的数据指定到染色体的相应基因位置
                job_num = job_data[j]  # 任务编号
                population[i][j][0] = job_num

                # 随机选择各个任务的一个加工机器，并指定到染色体的相应基因位置
                procedure_show_times = np.sum(job_data[0:j + 1] == job_num)  # 某任务编号是第几次出现，从而得出这是该任务的第几道工序
                population[i][j][1] = choice(list(filter(lambda x: x in self.machine_id_list,
                                                         self.jobs.loc[job_num].machines[
                                                             procedure_show_times - 1])))  # 从该任务的工序对应的可选机器列表中随机选择一个

        return population

    '''解码：①调度表，②目标值'''

    def decodeJobStrategy1(self, is_first_job_of_machine, machine_id, job_id):
        """对某机器上的任务进行解码"""

        # 数据准备
        '''结果存储，生成dataframe'''
        start_time_list = []
        end_time_list = []
        machine_list = []
        machine_status_list = []
        procedure_list = []
        category_id_list = []
        order_id_list = []
        product_id_list = []
        num_list = []

        '''结果存储，生成用于计算目标值的schedule_dataframe'''
        start_time_list_for_cal = []
        end_time_list_for_cal = []
        machine_list_for_cal = []
        machine_status_list_for_cal = []
        procedure_list_for_cal = []
        category_id_list_for_cal = []
        order_id_list_for_cal = []
        product_id_list_for_cal = []

        this_job = self.jobs.loc[job_id]
        this_machine = self.machines.loc[machine_id]
        order_batch = this_job.orders
        this_procedure = this_machine.procedure
        maintenance_day0 = this_machine.maintenance_day[0]  # 机器当月的维修日期
        maintenance_day1 = this_machine.maintenance_day[1]  # 机器下月的维修日期
        model_change_time = this_machine.model_change_time
        processing_time = this_machine.processing_time.loc[job_id]

        def linesAppend(status, start_time, end_time, category_ID, order_ID, product_ID, num_index):
            """往结果中增添一行"""
            if isinstance(start_time, list):
                num = len(start_time)
            else:
                num = 1
                start_time = [start_time]
                end_time = [end_time]
                num_index = [num_index]

            status = [status] * num
            category_ID = [category_ID] * num
            order_ID = [order_ID] * num
            product_ID = [product_ID] * num
            machine = [machine_id] * num
            procedure = [this_procedure] * num

            start_time_list.extend(start_time)
            end_time_list.extend(end_time)
            machine_list.extend(machine)
            machine_status_list.extend(status)
            procedure_list.extend(procedure)
            category_id_list.extend(category_ID)
            order_id_list.extend(order_ID)
            product_id_list.extend(product_ID)
            num_list.extend(num_index)

        def linesAppendForCal(status, start_time, end_time, category_ID, order_ID, product_ID):
            """往结果中增添一行"""
            start_time_list_for_cal.append(start_time)
            end_time_list_for_cal.append(end_time)
            machine_list_for_cal.append(machine_id)
            machine_status_list_for_cal.append(status)
            procedure_list_for_cal.append(this_procedure)
            category_id_list_for_cal.append(category_ID)
            order_id_list_for_cal.append(order_ID)
            product_id_list_for_cal.append(product_ID)

        def saveSegment(start_time, end_time, status, this_order, product_ID):
            def getItemStartEndTime(index):
                item_start_time_ = start_time + index * processing_time
                item_end_time_ = start_time + (index + 1) * processing_time
                return item_start_time_, item_end_time_

            if status == "生产":
                this_order_id = this_order.order_id
                item_index = list(range(this_order.product_number))
                item_start_end_time = list(map(getItemStartEndTime, item_index))
                item_start_time, item_end_time = zip(*item_start_end_time)
                item_start_time = list(item_start_time)
                item_end_time = list(item_end_time)
                linesAppend("生产", item_start_time, item_end_time, job_id, this_order_id, product_ID, item_index)
                linesAppendForCal('生产', start_time, end_time, job_id, this_order_id, product_ID)
            elif status == "维保":
                linesAppend('维保', start_time, end_time, '维保', '', '', '')
                linesAppendForCal('维保', start_time, end_time, '维保', '', '')
            elif status == "停机":
                linesAppend('停机', start_time, end_time, '停机', '', '', '')
                linesAppendForCal('停机', start_time, end_time, '停机', '', '')
            elif status == "空转":
                linesAppend('空转', start_time, end_time, '空转', '', '', '')
                linesAppendForCal('空转', start_time, end_time, '空转', '', '')

        model_index = 0
        for model_id, order_by_model in order_batch.items():  # 遍历该类别下的每个型号
            product_id = model_id
            for order_index, order in enumerate(order_by_model):  # 遍历型号对应的每个订单
                item_num = order.product_number
                order_processing_time = item_num * processing_time
                duration_start_time = 0
                duration_end_time = 0
                maintenance_start_time = 0
                maintenance_end_time = 0
                machine_status = 0

                # 计算订单的开始加工时间
                if (is_first_job_of_machine == 1) & (model_index == 0) & (order_index == 0):  # 若是机器上第一个job的第一个订单
                    order_start_time = this_job.latest_end_time  # 该job的上一道工序结束时间
                elif (model_index == 0) & (order_index == 0):  # 若是机器上非第一个job的第一个订单，需要换型
                    change_start_time = this_machine.latest_end_time
                    change_end_time = change_start_time + model_change_time
                    order_start_time = max(change_end_time,
                                           this_job.latest_end_time)  # 该job的上一道工序结束时间和机器上一次加工结束时间之间的最大值
                    # 判断两个job的生产间隔中机器的状态
                    if this_machine.is_shutdown_when_change_model:  # 如果换型需要停机，则机器停机
                        machine_status = "停机"
                    elif order_start_time - this_machine.latest_end_time > this_machine.max_idle_time:
                        # 如果换型不需停机，则判断间隔时间是否超过机器最大空转时间，如果超过，则机器停机；否则机器空转
                        machine_status = "停机"
                    else:
                        machine_status = "空转"
                    duration_start_time = this_machine.latest_end_time  # 两个job间隔时间段的起点
                    duration_end_time = order_start_time  # 两个job间隔时间段的终点
                else:
                    order_start_time = this_machine.latest_end_time + model_change_time
                    # 判断两个job的生产间隔中机器的状态
                    if this_machine.is_shutdown_when_change_model:  # 如果换型需要停机，则机器停机
                        machine_status = "停机"
                    elif order_start_time - this_machine.latest_end_time > this_machine.max_idle_time:
                        # 如果换型不需停机，则判断间隔时间是否超过机器最大空转时间，如果超过，则机器停机；否则机器空转
                        machine_status = "停机"
                    else:
                        machine_status = "空转"
                    duration_start_time = this_machine.latest_end_time  # 两个job间隔时间段的起点
                    duration_end_time = order_start_time  # 两个job间隔时间段的终点

                # 判断产品的加工开始时间是否在维保期
                if maintenance_day0 <= order_start_time <= maintenance_day0 + np.timedelta64(24, 'h'):
                    machine_status = "维保"
                    maintenance_start_time = maintenance_day0
                    maintenance_end_time = maintenance_day0 + np.timedelta64(24, 'h')
                    order_start_time = maintenance_end_time  # 在维保期后才能开始生产
                if maintenance_day1 <= order_start_time <= maintenance_day1 + np.timedelta64(24, 'h'):
                    machine_status = "维保"
                    maintenance_start_time = maintenance_day1
                    maintenance_end_time = maintenance_day1 + np.timedelta64(24, 'h')
                    order_start_time = maintenance_end_time  # 在维保期后才能开始生产

                # 如果该机器只有在人工的非夜班的时候才能生产
                if this_machine.no_work_at_night == 1:
                    try:
                        order_start_time = max(order_start_time, np.datetime64(
                            str(order_start_time.astype(object).strftime('%Y-%m-%d')) + ' 08:00'))
                    except AttributeError:
                        order_start_time = np.datetime64(order_start_time.strftime('%Y-%m-%d %H:%M:%S'))
                        order_start_time = max(order_start_time, np.datetime64(
                            str(order_start_time.astype(object).strftime('%Y-%m-%d')) + ' 08:00'))

                # 如果该机器只有在能耗的夜班才能生产
                if this_machine.work_at_night == 1:
                    try:
                        order_start_time = np.datetime64(order_start_time.strftime('%Y-%m-%d %H:%M:%S'))
                    except AttributeError:
                        pass
                    if (order_start_time <= np.datetime64(order_start_time.astype(object).strftime('%Y-%m-%d') + ' 06:00')) & \
                            (order_start_time >= np.datetime64(order_start_time.astype(object).strftime('%Y-%m-%d') + ' 22:00')):
                        pass
                    else:
                        order_start_time = np.datetime64(order_start_time.astype(object).strftime('%Y-%m-%d') + ' 22:00')

                # 该机器的第一次开机时间
                if (is_first_job_of_machine == 1) & (model_index == 0) & (order_index == 0):
                    this_machine.first_start_time = order_start_time

                # 计算订单的结束订单时间
                order_end_time = order_start_time + order_processing_time
                if machine_status == "维保":
                    saveSegment(maintenance_start_time, maintenance_end_time, "维保", order, product_id)
                    if not isinstance(duration_start_time, int):
                        saveSegment(duration_start_time, duration_end_time, "停机", order, product_id)
                elif not isinstance(duration_start_time, int):
                    saveSegment(duration_start_time, duration_end_time, machine_status, order, product_id)
                saveSegment(order_start_time, order_end_time, "生产", order, product_id)

                this_machine.latest_end_time = order_end_time
                this_job.latest_end_time = order_end_time

            model_index += 1

        result_df = pd.DataFrame(list(
            zip(machine_list, start_time_list, end_time_list, machine_status_list, procedure_list, category_id_list,
                order_id_list,
                product_id_list,
                num_list)))
        result_df_for_cal = pd.DataFrame(list(
            zip(machine_list_for_cal, start_time_list_for_cal, end_time_list_for_cal, machine_status_list_for_cal,
                procedure_list_for_cal,
                category_id_list_for_cal,
                order_id_list_for_cal,
                product_id_list_for_cal)))
        result_df.columns = ['Machine', 'Start Time', 'End Time', 'Machine Status', 'Procedure', 'Category ID',
                             'Order ID',
                             'Product ID', '#']
        result_df_for_cal.columns = ['Machine', 'Start Time', 'End Time', 'Machine Status', 'Procedure', 'Category ID',
                                     'Order ID',
                                     'Product ID']
        return result_df, result_df_for_cal

    def decodeChromosomeStrategy1(self, chromosome):
        """对染色体进行解码，返回调度表，以及机器第一次开机时间"""

        # 数据准备
        schedule_df = pd.DataFrame(
            columns=['Machine', 'Start Time', 'End Time', 'Machine Status', 'Procedure', 'Category ID', 'Order ID',
                     'Product ID',
                     '#'])  # dataframe格式的结果
        schedule_df_for_cal = pd.DataFrame(
            columns=['Machine', 'Start Time', 'End Time', 'Machine Status', 'Procedure', 'Category ID', 'Order ID',
                     'Product ID'])  # dataframe格式的结果
        job_count_of_machine = pd.Series([0] * self.machine_num, index=self.machine_id_list)  # 用于计算某机器上job的累计个数
        machine_first_start_time = []  # 机器第一次开机时间

        for i in range(self.chromosome_size):  # 遍历染色体上的所有基因
            job_id = chromosome[i][0]  # 任务编号
            machine_id = chromosome[i][1]  # 机器编号
            job_count_of_machine.loc[machine_id] += 1
            if job_count_of_machine.loc[machine_id] == 1:  # 该job是这个机器上的第一个加工任务
                result_df, result_df_for_cal = self.decodeJobStrategy1(1, machine_id, job_id)
            else:
                result_df, result_df_for_cal = self.decodeJobStrategy1(0, machine_id, job_id)
            schedule_df = pd.concat([schedule_df, result_df])
            schedule_df_for_cal = pd.concat([schedule_df_for_cal, result_df_for_cal])

        for machine in self.machines:
            machine_first_start_time.append(machine.first_start_time)

        self.data.resetMachine(self.machines)
        self.data.resetJob(self.jobs)
        schedule_df.index = [_ for _ in range(schedule_df.shape[0])]
        schedule_df_for_cal.index = [_ for _ in range(schedule_df_for_cal.shape[0])]
        return schedule_df, schedule_df_for_cal, machine_first_start_time

    def getObjectiveValue(self, chromosome):
        """获取个体的目标值"""
        schedule_dataframe, schedule_dataframe_for_cal, machine_first_start_time = self.decodeChromosomeStrategy1(
            chromosome)
        project_end_time = schedule_dataframe_for_cal['End Time'].max()
        energy_cost = self.cal.getEnergyCost(schedule_dataframe_for_cal, machine_first_start_time)
        labor_cost = self.cal.getLaborCost(schedule_dataframe_for_cal)
        piece_cost = self.cal.piece_cost
        hold_cost = self.cal.getHoldCost(schedule_dataframe)
        delay_cost = self.cal.getDelayCost(schedule_dataframe_for_cal)
        objective_value = energy_cost + labor_cost + piece_cost + hold_cost + delay_cost
        return project_end_time, objective_value, [energy_cost, labor_cost, piece_cost, hold_cost, delay_cost]

    def getFitness(self, query_population):
        """获取种群的适应度，返回适应度的array"""
        fitness_list = []
        project_end_time_list = []
        cost_list = []
        for i in range(len(query_population)):  # 遍历种群中的每个个体
            # 对于一个新的个体，就要重置jobs和machines（latest end time，first start time），实例对象属性绑定
            project_end_time, objective_value, costs = self.getObjectiveValue(query_population[i])
            fitness = 1 / objective_value
            fitness_list.append(fitness)
            project_end_time_list.append(project_end_time)
            cost_list.append(costs)
        fitness_array = np.array(fitness_list)
        return project_end_time_list, fitness_array, cost_list

    def getBestChromosome(self, population, fitness_array, project_end_time_list, cost_list, best_rate):
        """获取种群中若干个的最好个体，最优目标值，最优结束时间和最优调度表"""
        sort_index = np.argsort(-fitness_array)
        if best_rate >= 1:
            best_num = best_rate
        else:
            best_num = int(self.population_size * best_rate)
        best_chromosomes = population[sort_index][:best_num]
        best_objective_value = np.reciprocal(fitness_array[sort_index][:best_num])
        best_end_time = np.array(project_end_time_list)[sort_index][:best_num]
        best_cost = np.array(cost_list)[sort_index][:best_num]

        return best_chromosomes, best_objective_value, best_end_time, best_cost

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
                chromosome[j][1] = choice(
                    list(
                        filter(
                            lambda x: x in self.machine_id_list, self.jobs.loc[job_id].machines[process_index])))

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
            project_end_time_list, fitness_array, cost_list = self.getFitness(offsprings)
            best_chromosome, _, _, _ = self.getBestChromosome(offsprings, fitness_array, project_end_time_list,
                                                              cost_list, 1)
            return best_chromosome

    def mutationPopulation(self, population, iterate_num):
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

    def select(self, offspring_population, best_chromosome, fitness_array, select_rate):
        """选择用于交叉变异的个体"""
        sort_index = np.argsort(-fitness_array)
        select_num = int((self.population_size - len(best_chromosome)) * select_rate)
        select_population = offspring_population[sort_index][:select_num]
        new_num = self.population_size - select_num - len(best_chromosome)
        new_population = self.initialPopulation(new_num)
        return np.vstack((select_population, best_chromosome, new_population))

    '''结果输出'''

    def resultExport(self, schedule, fitness, costs, obj_val, end):

        def exportSchedule():
            """调度表"""
            schedule_path = "调度表.xlsx"
            if save_file:
                schedule.to_excel(result_folder_path + '\\' + schedule_path)

        def plotFitnessEvolution():
            """目标值迭代曲线"""
            evolution_path = "进化.png"
            # 准备用于作图的数据
            y_total = fitness
            energy_cost, labor_cost, piece_cost, hold_cost, delay_cost = zip(*costs)
            y_energy = list(energy_cost)
            y_labor = list(labor_cost)
            y_hold = list(hold_cost)
            y_delay = list(delay_cost)
            len_y = len(y_total)
            x_value = [i for i in range(1, len_y + 1)]

            plt.figure(figsize=[18, 9], constrained_layout=True)
            ax_total = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
            ax_energy = plt.subplot2grid((2, 4), (0, 2))
            ax_labor = plt.subplot2grid((2, 4), (0, 3))
            ax_hold = plt.subplot2grid((2, 4), (1, 2))
            ax_delay = plt.subplot2grid((2, 4), (1, 3))
            ax_total.plot(x_value, y_total)
            ax_energy.plot(x_value, y_energy)
            ax_labor.plot(x_value, y_labor)
            ax_hold.plot(x_value, y_hold)
            ax_delay.plot(x_value, y_delay)
            ax_total.set_title("总成本", fontsize=14)
            ax_energy.set_title("能耗成本", fontsize=14)
            ax_labor.set_title("人工成本", fontsize=14)
            ax_hold.set_title("库存成本", fontsize=14)
            ax_delay.set_title("延迟成本", fontsize=14)
            plt.suptitle("成本迭代曲线", fontsize=14)
            ax_total.tick_params(axis='x', labelsize=12)
            ax_total.tick_params(axis='y', labelsize=12)
            ax_total.set_xlabel(u"迭代次数", fontsize=14)
            ax_total.set_ylabel(u"成本", fontsize=14)
            ax_energy.tick_params(axis='x', labelsize=12)
            ax_energy.tick_params(axis='y', labelsize=12)
            ax_energy.set_xlabel(u"迭代次数", fontsize=14)
            ax_energy.set_ylabel(u"成本", fontsize=14)
            ax_labor.tick_params(axis='x', labelsize=12)
            ax_labor.tick_params(axis='y', labelsize=12)
            ax_labor.set_xlabel(u"迭代次数", fontsize=14)
            ax_labor.set_ylabel(u"成本", fontsize=14)
            ax_hold.tick_params(axis='x', labelsize=12)
            ax_hold.tick_params(axis='y', labelsize=12)
            ax_hold.set_xlabel(u"迭代次数", fontsize=14)
            ax_hold.set_ylabel(u"成本", fontsize=14)
            ax_delay.tick_params(axis='x', labelsize=12)
            ax_delay.tick_params(axis='y', labelsize=12)
            ax_delay.set_xlabel(u"迭代次数", fontsize=14)
            ax_delay.set_ylabel(u"成本", fontsize=14)
            plt.rcParams['axes.unicode_minus'] = False

            # 先保存再show，否则保存的图片可能是空白的
            plt.tight_layout()
            if save_file:
                plt.savefig(result_folder_path + '\\' + evolution_path, dpi=700)  # 算法、策略
            plt.show()

        def plotGantt():
            """甘特图"""
            gantt_png_path = "甘特.png"
            gantt_html_path = "甘特.html"

            # 对schedule进行调整
            schedule_reformat = schedule[
                (schedule['Machine Status'] == '生产') | (schedule['Machine Status'] == '空转')]
            machine_list = schedule_reformat.Machine.unique()
            machine_list = np.sort(machine_list)
            machine_list = machine_list[::-1]  # 转为降序
            machine_name_list = list(map(lambda x: 'M' + str(x), machine_list))
            schedule_reformat.loc[:, 'Machine'] = schedule_reformat.Machine.map(lambda machineid: "M" + str(machineid))
            fig = px.timeline(schedule_reformat, x_start='Start Time', x_end='End Time', y='Machine',
                              color='Category ID',
                              color_discrete_map={1: 'red', 2: 'green', 3: 'blue', 4: 'goldenrod', 5: 'magenta',
                                                  6: 'purple', 7: 'yellow', 8: 'pink', '空转': 'grey'},
                              hover_name='Order ID',
                              category_orders={'Machine': machine_name_list,
                                               'Category ID': [1, 2, 3, 4, 5, 6, 7, 8, '空转']},
                              labels={'Machine': '设备',
                                      'Category ID': '产品类别'})
            fig.update_yaxes(showgrid=True, griddash='dash')
            fig.add_annotation(text=f'目标值：{obj_val}，完工时间：{end}',
                               align='left',
                               showarrow=False,
                               xref='paper',
                               yref='paper',
                               x=1,
                               xanchor='right',
                               y=1,
                               yanchor='bottom',
                               bgcolor='white')
            fig.update_layout(font_size=16)
            if save_file:
                fig.write_image(result_folder_path + '\\' + gantt_png_path, width=2000, height=1000)
                plotly.offline.plot(fig, filename=result_folder_path + '\\' + gantt_html_path)
            fig.show()
            print(f"最终设备选择为：{machine_list[::-1]}")

        save_file = True
        obj_val = int(obj_val)
        end = end.strftime('%Y-%m-%d %H:%M')
        folder_path = "C:\\Users\\ejauxue002\\Nutstore\\1\\Q2\\实验结果记录"
        result_folder_path = folder_path + '\\' + self.file_name
        if os.path.exists(folder_path):
            save_file = True
            try:
                os.mkdir(result_folder_path)
            except FileExistsError:
                while os.path.exists(result_folder_path):
                    result_folder_path = result_folder_path + '_1'
                os.mkdir(result_folder_path)
        else:
            save_file = False
        exportSchedule()
        plotFitnessEvolution()
        plotGantt()

    '''多进程'''

    def decodeChildTask(self, chromosome):
        """用于多进程解码的实现，个体的解码函数"""
        project_end_time, objective_value, costs = self.getObjectiveValue(chromosome)
        print("\033[1;32m", "█", "\033[0m", sep="", end="")
        return project_end_time, 1 / objective_value, costs

    def multiprocessDecode(self, population, iterate_count):
        """多进程解码"""
        print("解码多进程开始进行：")
        decode_multiprocess = Multiprocess()
        decode_args = zip(population)
        decode_results = decode_multiprocess.work(self.decodeChildTask, decode_args, iterate_count)
        fitness_array = np.array(decode_results[1])  # 把适应度list转化为array
        return decode_results[0], fitness_array, decode_results[2]

    def mutationChildTask(self, population, iterate_num):
        """用于多进程变异的实现，种群变异函数"""
        mutation_offspring = self.mutationOperator(population, iterate_num)
        print("\033[1;32m", "█", "\033[0m", sep="", end="")
        return mutation_offspring

    def multiprocessMutation(self, population, iterate_count):
        print("变异多进程开始进行：")
        mutation_num = int(self.mutation_rate * self.population_size)
        mutation_multiprocess = Multiprocess()
        mutation_args = zip([population] * mutation_num, [iterate_count] * mutation_num)
        mutation_results = mutation_multiprocess.work(self.mutationChildTask, mutation_args, iterate_count)
        mutation_population = np.array(mutation_results[0])
        return mutation_population

    def mutation(self, population, iterate_num):
        """整合多进程和单进程的变异"""
        if iterate_num > self.mutation_change_point:
            mutation_offspring = self.mutationPopulation(population, iterate_num)
        else:
            mutation_offspring = self.multiprocessMutation(population, iterate_num)
        return mutation_offspring

    def execute(self, visual_flag=True):
        """执行GA"""
        obj_evolution = []  # 记录每一代的最优目标值
        cost_evolution = []  # 记录每一代的最优成本
        best_chromosome = []  # 记录当前的最优个体
        best_objective_value = 0  # 记录当前的最优目标值
        best_end = 0  # 记录当前的最优完工时间
        best_cost = []  # 记录当前的最优的各项成本
        execute_flag = True
        execute_count = 1
        iterate_count_all = 1  # 记录总的迭代次数
        best_keep_rate = self.best_keep_rate

        #############初始化种群############
        while execute_flag:
            another_execute = False
            iterate_flag = True
            iterate_count = 0

            ##############生成初始种群#############
            print(
                f"----------------------------------------------第{execute_count}次执行GA，第{iterate_count}代----------------------------------------------")
            population = self.initialPopulation(self.population_size)
            project_end_time, fitness_array, cost_list = self.multiprocessDecode(population, iterate_count)
            if isinstance(best_chromosome, np.ndarray):
                best_chromosome, best_objective_value, best_end, best_cost = self.getBestChromosome(
                    np.vstack((population, best_chromosome)),
                    np.append(fitness_array, 1 / best_objective_value, axis=0),
                    np.append(project_end_time, best_end, axis=0),
                    np.append(cost_list, best_cost, axis=0),
                    best_keep_rate)
            else:
                best_chromosome, best_objective_value, best_end, best_cost = self.getBestChromosome(population,
                                                                                                    fitness_array,
                                                                                                    project_end_time,
                                                                                                    cost_list,
                                                                                                    best_keep_rate)
            print("第%s代：最优个体的目标值为：%s，项目结束时间为：%s" % (
                iterate_count, best_objective_value[0], best_end[0]))
            iterate_count += 1

            #############进行若干次进化#############
            while iterate_flag:  # 进化次数
                print(
                    f"----------------------------------------------第{execute_count}次执行GA，第{iterate_count}代----------------------------------------------")

                select_rate = self.select_rate - (iterate_count_all - 1) / 4 / self.evolution_num
                best_keep_rate = self.best_keep_rate - (iterate_count_all - 1) / 4 / self.evolution_num

                ######################交叉和变异#####################
                crossover_population = self.crossOver(population)
                mutation_population = self.mutation(population, iterate_count)
                offspring_population = np.vstack((crossover_population, mutation_population))

                ####################种群适应度计算####################
                project_end_time, fitness_array, cost_list = self.multiprocessDecode(offspring_population,
                                                                                     iterate_count)

                ####################更新种群优秀个体###################
                best_chromosome, best_objective_value, best_end, best_cost = self.getBestChromosome(
                    np.vstack((offspring_population, best_chromosome)),
                    np.append(fitness_array, 1 / best_objective_value, axis=0),
                    np.append(project_end_time, best_end, axis=0),
                    np.append(cost_list, best_cost, axis=0),
                    best_keep_rate)
                # 记录本代的最优目标值
                obj_evolution.append(best_objective_value[0])
                cost_evolution.append(best_cost[0])
                print("第%s代：最优个体的目标值为：%s，项目结束时间为：%s" % (
                    iterate_count, best_objective_value[0], best_end[0]))

                ##################选择种群中的优质个体#################
                population = self.select(offspring_population, best_chromosome, fitness_array, select_rate)

                ########################结束判断#######################
                def isStop():

                    nonlocal iterate_flag, another_execute

                    if iterate_count_all > self.evolution_num:
                        print(f"迭代进化次数超过{self.evolution_num}，结束GA运行，输出最终结果。")
                        iterate_flag = False
                        another_execute = False
                    elif iterate_count_all > 6:
                        best_obj_1 = obj_evolution[iterate_count_all - 2]
                        best_obj_2 = obj_evolution[iterate_count_all - 3]
                        best_obj_3 = obj_evolution[iterate_count_all - 4]
                        best_obj_4 = obj_evolution[iterate_count_all - 5]
                        best_obj_5 = obj_evolution[iterate_count_all - 6]
                        # 计算此代的最优值与前5代最优值之间的平方差
                        MSE = pow(best_objective_value[0] - best_obj_1, 2) + \
                              pow(best_objective_value[0] - best_obj_2, 2) + \
                              pow(best_objective_value[0] - best_obj_3, 2) + \
                              pow(best_objective_value[0] - best_obj_4, 2) + \
                              pow(best_objective_value[0] - best_obj_5, 2)
                        if MSE < 0.05:  # 最优值6代内不再变化
                            print("最优值6代内不再变化，终止本次GA运行，开启一次新的GA运行。")
                            iterate_flag = False
                            another_execute = True

                    # 通过外界输入来控制进程
                    if iterate_count_all > 40:
                        key_board_input = input(
                            "按Enter结束GA运行，输出最终结果；按c终止本次GA运行，开启新一次的GA运行；否则继续运行：")
                        if key_board_input == '':
                            print("结束GA运行，输出最终结果。")
                            iterate_flag = False
                            another_execute = False
                        elif key_board_input == 'c':
                            print(f"终止第{execute_count}次GA运行，开启下一次GA运行。")
                            iterate_flag = False
                            another_execute = True
                        else:
                            print(f"继续本次的GA运行，进入第{iterate_count + 1}次迭代进化。")

                iterate_count += 1
                iterate_count_all += 1
                isStop()

            execute_flag = another_execute

            if not another_execute:
                """结果输出"""
                if visual_flag:
                    schedule_df, schedule_df_for_cal, machine_first_start_time = self.decodeChromosomeStrategy1(
                        best_chromosome[0])
                    best_energy_cost = self.cal.getEnergyCost(schedule_df_for_cal, machine_first_start_time)
                    best_labor_cost = self.cal.getLaborCost(schedule_df_for_cal)
                    best_piece_cost = self.cal.piece_cost
                    best_hold_cost = self.cal.getHoldCost(schedule_df)
                    best_delay_cost = self.cal.getDelayCost(schedule_df_for_cal)
                    self.resultExport(schedule_df_for_cal, obj_evolution, cost_evolution, best_objective_value[0],
                                      best_end[0])
                    print(
                        "-----------------------------------------------------------------------------------------------------------")
                    print("最好目标值为：" + str(best_objective_value[0]))
                    print("对应的能耗成本为：" + str(best_energy_cost))
                    print("对应的人工成本为：" + str(best_labor_cost))
                    print("对应的计件成本为：" + str(best_piece_cost))
                    print("对应的库存成本为：" + str(best_hold_cost))
                    print("对应的延迟成本为：" + str(best_delay_cost))
                    print("完工时间为：" + str(best_end[0]))

                return best_objective_value[0]

            execute_count += 1
