from interval import IntervalSet
from origindata import *

"""
更新说明：
修改全局变量DATA的使用范围，python中的全局变量只是模块级别的，不可跨模块访问，因此将其作为参数传入。
删掉dataGenerator()

"""
DATA = object()


def importGlobalData(data):
    """将其他模块生成的data传入此模块，并作为全局变量供整个模块使用"""
    global DATA
    DATA = data


class Product:
    """
    产品类：某个具体的单个产品，包含加工工序，加工机器，所属型号，所属类别
    属性不可任意修改
    """

    def __init__(self, model_id):
        self.category_id = DATA.category_to_model.loc[model_id, "产品类别"]  # 产品的类别编号
        self.model_id = model_id  # 产品的型号编号
        self.procedures = DATA.procedure_of_category[self.category_id - 1]  # 包含的加工工序
        self.machines = DATA.machine_of_category[self.category_id - 1]  # 各道工序的加工机器


class CategoryProduct:
    """
    产品抽象类：表示某个产品类别，包含加工工序，加工机器
    除latest_end_time外的属性不可修改
    """

    def __init__(self, category_id):
        self.category_id = category_id
        self.procedures = DATA.procedure_of_category[category_id - 1]
        self.machines = DATA.machine_of_category[category_id - 1]
        self.procedures = DATA.procedure_of_category[category_id - 1]
        self.latest_end_time = 0  # 产品类别的最近的加工结束时间，外部可以修改，设置重置函数


class Machine:
    """
    机器类：某个具体的机器，包含设备编号，加工工序，对应产品类别，换型时间，换型是否停机，设备作业时间，设备能耗，设备维保时间
    除first_start_time和latest_end_time外的属性不可随意修改
    """

    def __init__(self, machine_id):
        self.machine_id = machine_id  # 机器编号
        self.procedure = DATA.machine_information.loc[machine_id, "对应工序"]
        self.product_category = DATA.machine_information.loc[machine_id, "对应产品类别"]
        self.model_change_time = DATA.machine_information.loc[machine_id, "换型时间"]
        self.is_shutdown_when_change_model = DATA.machine_information.loc[machine_id, "换型是否停机"]
        self.maintenance_day = DATA.machine_information.loc[machine_id, "维保日期"]
        self.processing_time = DATA.processing_time.loc[machine_id]
        self.energy_consume_produce = DATA.machine_energy_consumption.loc[machine_id, "生产能耗/小时"]
        self.energy_consume_idle = DATA.machine_energy_consumption.loc[machine_id, "空转(开机等待)能耗/小时"]
        self.energy_consume_startup = DATA.machine_energy_consumption.loc[machine_id, "开机一次性能耗"]
        self.max_idle_time = DATA.machine_energy_consumption.loc[machine_id, "生产间隔时间超过则停机小时数"]
        self.first_start_time = 0  # 首次开机时间，外部可以修改的属性，设置重置函数
        self.latest_end_time = 0  # 最近一次的加工结束时间，外部可以修改的属性，设置重置函数


class Order:
    """
    订单类：某个具体的订单，包括订单编号，订单日期，期望交期，产品型号，数量。
    其属性不能任意修改。
    """

    def __init__(self, order_id):
        self.order_id = order_id
        self.order_date = DATA.order_table.loc[order_id, "订单日期"]
        self.due_date = DATA.order_table.loc[order_id, "期望交期"]
        self.product_model = DATA.order_table.loc[order_id, "产品型号"]
        self.product_number = DATA.order_table.loc[order_id, "数量"]
        self.product_category = DATA.order_table.loc[order_id, "产品类别"]
        self.lead_time = DATA.lead_time


class Price:
    """成本价格类，包括能耗价格，人工价格等等，后续补充"""

    def __init__(self):
        self.energy_price_day = DATA.energy_price['白班'][0]
        self.energy_price_evening = DATA.energy_price['晚班'][0]
        self.energy_price_night = DATA.energy_price['夜班'][0]
        self.energy_price = DATA.energy_price

    def getDateFactor(self, date):
        """获取某日期的日期价格因子"""
        date_factor = 0
        date_factor_series = self.energy_price[self.energy_price['日期'] == date]['日期价格因子']
        date_factor_list = date_factor_series.tolist()
        try:
            date_factor = date_factor_list[0]
        except IndexError:
            print("IndexError")
            print(date)
        return date_factor


class CalculateUtils:
    """
    计算相关的工具包：目标函数计算
    """

    def __init__(self):
        self.DATA = DATA
        self.price = Price()
        self.day_shift = IntervalSet.between("06:00", "16:00")  # 白班
        self.evening_shift = IntervalSet.between("16:00", "22:00")  # 晚班
        self.night_shift0 = IntervalSet.between("22:00", "23:59")  # 夜班
        self.night_shift1 = IntervalSet.between("00:00", "06:00")  # 夜班

    def getEnergyCost(self, schedule, machine_first_start_time):
        """获得某个调度表下的能耗成本"""

        def getMachineEnergyConsume(Id, status):
            """对外接口，获取某个机器在某状态下的能耗"""
            if status == '生产':
                return self.DATA.machine_energy_consumption.loc[Id, "生产能耗/小时"]
            elif status == '空转':
                return self.DATA.machine_energy_consumption.loc[Id, "空转(开机等待)能耗/小时"]
            else:
                return self.DATA.machine_energy_consumption.loc[Id, "开机一次性能耗"]

        def getEnergyPrice(duration_start_time, duration_end_time):
            """获取当前时间段下的能耗价格*时长；如果两个时间相等，就是获取当前时间的能耗价格"""

            def get_overlap_hours(query_duration, shift):
                """获取某时间段在某个班次下的时长"""
                overlap_duration = query_duration & shift
                if overlap_duration:
                    upper = overlap_duration.upper_bound()
                    lower = overlap_duration.lower_bound()
                    duration_length = datetime.datetime.strptime(upper, "%H:%M") - \
                                      datetime.datetime.strptime(lower, "%H:%M")
                    if upper == '23:59':
                        hours = (duration_length.total_seconds() + 60) / 3600  # 加上23:59到0:00之间的1分钟
                    else:
                        hours = duration_length.total_seconds() / 3600
                else:
                    hours = 0

                return hours

            if duration_start_time == 0:  # 意味着这个机器没有启用
                return 0

            try:
                duration_start_time = duration_start_time.astype(datetime.datetime)
                duration_end_time = duration_end_time.astype(datetime.datetime)
            except AttributeError:
                duration_start_time = datetime.datetime.strptime(duration_start_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
                duration_end_time = datetime.datetime.strptime(duration_end_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
            start_date = duration_start_time.date()
            end_date = duration_end_time.date()
            duration_start_time = duration_start_time.strftime("%H:%M")
            duration_end_time = duration_end_time.strftime("%H:%M")

            if duration_start_time == duration_end_time:
                date_factor = self.price.getDateFactor(start_date)
                if duration_start_time in self.day_shift:
                    return self.price.energy_price_day * date_factor
                elif duration_start_time in self.evening_shift:
                    return self.price.energy_price_evening * date_factor
                else:
                    return self.price.energy_price_night * date_factor
            elif start_date == end_date:
                duration = IntervalSet.between(duration_start_time, duration_end_time)  # 当前时间段
                date_factor = self.price.getDateFactor(start_date)
                day_hours = get_overlap_hours(duration, self.day_shift)
                evening_hours = get_overlap_hours(duration, self.evening_shift)
                night_hours = get_overlap_hours(duration, self.night_shift0) + get_overlap_hours(duration, self.night_shift1)
                price = day_hours * self.price.energy_price_day + evening_hours * self.price.energy_price_evening + night_hours * self.price.energy_price_night
                total_price = price * date_factor
                return total_price
            else:
                date_factor0 = self.price.getDateFactor(start_date)  # 第一天
                date_factor1 = self.price.getDateFactor(end_date)  # 第二天
                duration0 = IntervalSet.between(duration_start_time, "23:59")
                duration1 = IntervalSet.between("00:00", duration_end_time)
                '''由于一个duration（生产或空转）不会超过三小时，为了减少计算时间，只计算以下班次的重叠部分'''
                evening_hours = get_overlap_hours(duration0, self.evening_shift)
                night_hours0 = get_overlap_hours(duration0, self.night_shift0)
                night_hours1 = get_overlap_hours(duration1, self.night_shift1)  # 第二天夜班的重叠时长
                price0 = evening_hours * self.price.energy_price_evening + night_hours0 * self.price.energy_price_night  # 第一天
                price1 = night_hours1 * self.price.energy_price_night  # 第二天
                total_price = price0 * date_factor0 + price1 * date_factor1
                return total_price

        total_energy_cost = 0
        machine_first_start_price = np.zeros(shape=self.DATA.machine_num)  # 机器第一次开机时的能耗价格

        for _, row in schedule.iterrows():
            machine_id = row['Machine']
            machine_status = row['Machine Status']
            start_time = row['Start Time']
            end_time = row['End Time']
            energy_consumption = getMachineEnergyConsume(machine_id, machine_status)
            if machine_status == "停机":
                energy_price = getEnergyPrice(end_time, end_time)
            else:
                energy_price = getEnergyPrice(start_time, end_time)
            total_energy_cost += energy_consumption * energy_price

        # 计算机器第一次开机时的能耗成本
        for i, first_start_time in enumerate(machine_first_start_time):
            machine_first_start_price[i] = getEnergyPrice(first_start_time, first_start_time)
        first_start_energy_cost = self.DATA.startup_energy_consumption * machine_first_start_price

        total_energy_cost += first_start_energy_cost.sum()
        return total_energy_cost

    @staticmethod
    def time_to_str(t):
        """把python的时间（如datetime64、Timestamp）转化为字符串"""
        if isinstance(t, np.datetime64):
            t = t.astype(datetime.datetime)
        try:
            return t.strftime('%Y%m%d%H%M%S')
        except ValueError:
            print(ValueError)


class GAData:
    """遗传算法数据类"""

    def __init__(self):
        self.DATA = DATA
        self.chromosome_size = sum(DATA.procedure_num_of_category)
        self.category_list = list(set(DATA.order_table["产品类别"]))
        self.job_num = len(self.category_list)
        self.jobs = self.getJobs()
        self.procedure_num = DATA.procedure_num_of_category
        self.machine_num = DATA.machine_num
        self.machine_list = DATA.machine_list
        self.machines = self.getMachines()
        self.order_id_list = list(DATA.order_table.index)
        self.order_batches = self.getOrderBatches()
        self.setJobMachineLatestEndTime()

    def getJobs(self):
        """获取所有的任务对象"""
        jobs = []
        for category_id in self.category_list:
            job = CategoryProduct(category_id)
            jobs.append(job)
        jobs = pd.Series(jobs)
        jobs.index = self.category_list
        return jobs

    def getMachines(self):
        """获取所有的机器对象"""
        machines = []
        for machine_id in self.machine_list:
            machine = Machine(machine_id)
            machines.append(machine)
        machines = pd.Series(machines)
        machines.index = self.machine_list
        return machines

    def getOrderBatches(self):
        """按照类别对订单进行捆绑处理"""

        def getOrderByCategoryAndModel(category_id, model_id):
            order_list = []
            order_id_tuple = self.DATA.order_table[(self.DATA.order_table["产品类别"] == category_id) &
                                              (self.DATA.order_table["产品型号"] == model_id)
                                              ].index
            for order_id in order_id_tuple:
                order = Order(order_id)
                order_list.append(order)
            return order_list

        order_table_group = self.DATA.order_table.groupby(["产品类别", "产品型号"]).sum()
        multi_index = order_table_group.index
        order_batches = pd.Series(index=multi_index, dtype='object')

        for x in multi_index:
            this_category_id = x[0]
            this_model_id = x[1]
            order_batches.loc[this_category_id, this_model_id] = getOrderByCategoryAndModel(this_category_id,
                                                                                            this_model_id)
        return order_batches

    def setJobMachineLatestEndTime(self):
        """根据订单信息设置job的最早开始加工时间"""
        for job in self.jobs:
            job.latest_end_time = self.DATA.order_earliest_start_time

        for machine in self.machines:
            machine.latest_end_time = self.DATA.order_earliest_start_time

    def resetMachine(self, machines):
        """每遍历一个个体，重置一次，否则这些属性值会进入下一次迭代"""
        for machine in machines:
            machine.first_start_time = 0
            try:
                machine.latest_end_time = self.DATA.order_earliest_start_time
            except AttributeError:
                print(type(self.DATA))

    def resetJob(self, jobs):
        for job in jobs:
            job.latest_end_time = self.DATA.order_earliest_start_time


def getGa_data(order_name, order_time):
    """生成遗传算法对象以及数据准备"""
    origin_data = Data("题目2.dataset-v2.xlsx", order_name, order_time)  # 对全局变量origin_data进行修改
    importGlobalData(origin_data)  # 把origin_data导入datav20221110模块，一次性的
    ga_data = GAData()
    return ga_data
