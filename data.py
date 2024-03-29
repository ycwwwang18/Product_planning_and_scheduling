import copy
from random import choice

import numpy as np
import pandas as pd
from interval import IntervalSet
from origindata import *

DATA = object()


def importGlobalData(data):
    """将其他模块生成的data传入此模块，并作为全局变量供整个模块使用"""
    global DATA
    DATA = data


class Job:
    """
    任务类：以类别为单位的任务或以型号为单位的加工任务
    属性不可任意修改
    """

    def __init__(self, category_id=0, model_id=0):
        if category_id:
            self.category_id = category_id
            self.orders = self.getOrdersByCategory(category_id)  # 该任务的订单内容
        else:
            self.category_id = DATA.category_to_model.loc[model_id, "产品类别"]  # 产品的类别编号
            self.model_id = model_id  # 产品的型号编号
            self.orders = self.getOrdersByModel(model_id)
        self.machines = DATA.machine_of_category[self.category_id - 1]
        self.procedures = DATA.procedure_of_category[self.category_id - 1]  # 包含的加工工序
        self.procedure_num = len(self.procedures)
        self.feasible_machines = DATA.machine_of_category[self.category_id - 1]  # 各道工序的可行加工机器
        self.earliest_start_time = DATA.order_earliest_start_time
        self.latest_end_time = self.earliest_start_time  # 任务的最近加工结束时间，外部可以修改，设置重置函数

    def reset(self):
        self.latest_end_time = self.earliest_start_time

    @staticmethod
    def getOrdersByCategory(category_id_):
        """按照类别对订单进行捆绑处理"""
        order_df = DATA.order_table[(DATA.order_table["产品类别"] == category_id_)]
        model_id_set = set(order_df["产品型号"])
        order_batches = pd.Series(index=model_id_set, dtype=object)
        for model_id in model_id_set:
            order_id_by_model = order_df[order_df["产品型号"] == model_id].index
            order_list = []
            for order_id in order_id_by_model:
                order = Order(order_id)
                order_list.append(order)
            order_batches.loc[model_id] = order_list
        return order_batches

    @staticmethod
    def getOrdersByModel(model_id_):
        order_df = DATA.order_table[(DATA.order_table["产品型号"] == model_id_)]
        order_id_by_model = order_df.index
        order_list = []
        for order_id in order_id_by_model:
            order = Order(order_id)
            order_list.append(order)
        return order_list


class CategoryProduct:
    """
    产品抽象类：表示某个产品类别，包含加工工序，加工机器
    除latest_end_time外的属性不可修改
    """

    def __init__(self, category_id):
        self.category_id = category_id
        self.procedures = DATA.procedure_of_category[category_id - 1]
        self.machines = DATA.machine_of_category[category_id - 1]
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
        self.earliest_start_time = DATA.order_earliest_start_time
        self.first_start_time = 0  # 首次开机时间，外部可以修改的属性，设置重置函数
        self.latest_end_time = self.earliest_start_time  # 最近一次的加工结束时间，外部可以修改的属性，设置重置函数
        self.no_work_at_night = 0  # 该机器不在人工的夜班加工
        self.work_at_night = 0  # 该机器只能在能耗的夜班加工

    def reset(self):
        self.first_start_time = 0
        self.latest_end_time = self.earliest_start_time


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
        self.labor_price_day = DATA.cost_for_procedure['白班'][1]
        self.labor_price_evening = DATA.cost_for_procedure['晚班'][1]
        self.labor_price_night = DATA.cost_for_procedure['夜班'][1]
        self.hold_price = 100  # 库存成本为100/天
        self.delay_price = 100  # 延迟成本为100/天

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
        self.day_shift_labor = IntervalSet.between("08:00", "16:00")  # 人工白班
        self.evening_shift_labor = IntervalSet.between("16:00", "23:59")  # 人工晚班
        self.night_shift_labor = IntervalSet.between("00:00", "08:00")  # 人工夜班
        self.piece_cost = DATA.piece_cost

    def getMachineEnergyConsume(self, Id, status):
        """获取某个机器在某状态下的能耗"""
        if status == '生产':
            return self.DATA.machine_energy_consumption.loc[Id, "生产能耗/小时"]
        elif status == '空转':
            return self.DATA.machine_energy_consumption.loc[Id, "空转(开机等待)能耗/小时"]
        else:
            return self.DATA.machine_energy_consumption.loc[Id, "开机一次性能耗"]

    @staticmethod
    def getOverlapHours(query_duration, shift):
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

    def getLaborHours(self, duration_start_time, duration_end_time, shift_hour_dict):
        # 转化时间为datetime.datetime类型
        try:
            duration_start_time = duration_start_time.astype(datetime.datetime)
            duration_end_time = duration_end_time.astype(datetime.datetime)
        except AttributeError:
            duration_start_time = datetime.datetime.strptime(duration_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                                                             '%Y-%m-%d %H:%M:%S')
            duration_end_time = datetime.datetime.strptime(duration_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                                                           '%Y-%m-%d %H:%M:%S')
        start_date = duration_start_time.date()
        end_date = duration_end_time.date()

        if start_date == end_date:  # 该时间段在一天内
            duration_start_time = duration_start_time.strftime("%H:%M")
            duration_end_time = duration_end_time.strftime("%H:%M")
            duration = IntervalSet.between(duration_start_time, duration_end_time)  # 当前时间段
            day_hours = self.getOverlapHours(duration, self.day_shift_labor)
            evening_hours = self.getOverlapHours(duration, self.evening_shift_labor)
            night_hours = self.getOverlapHours(duration, self.night_shift_labor)
            if start_date in shift_hour_dict:
                shift_hour_dict[start_date]["day"] += day_hours
                shift_hour_dict[start_date]["evening"] += evening_hours
                shift_hour_dict[start_date]["night"] += night_hours
            else:
                shift_hour_dict[start_date] = {"day": day_hours,
                                               "evening": evening_hours,
                                               "night": night_hours}
        else:  # 该时间段跨了多天
            duration_start_time = duration_start_time.strftime("%H:%M")
            duration_end_time = duration_end_time.strftime("%H:%M")
            date = start_date
            while date <= end_date:
                if date == start_date:
                    duration = IntervalSet.between(duration_start_time, "23:59")
                elif date == end_date:
                    duration = IntervalSet.between("00:00", duration_end_time)
                else:
                    duration = IntervalSet.between("00:00", "23:59")
                day_hours = self.getOverlapHours(duration, self.day_shift_labor)
                evening_hours = self.getOverlapHours(duration, self.evening_shift_labor)
                night_hours = self.getOverlapHours(duration, self.night_shift_labor)
                if date in shift_hour_dict:
                    shift_hour_dict[date]["day"] += day_hours
                    shift_hour_dict[date]["evening"] += evening_hours
                    shift_hour_dict[date]["night"] += night_hours
                else:
                    shift_hour_dict[date] = {"day": day_hours,
                                             "evening": evening_hours,
                                             "night": night_hours}
                date += datetime.timedelta(days=1)
        return shift_hour_dict

    def getEnergyPrice(self, duration_start_time, duration_end_time):
        """获取当前时间段下的能耗价格*时长；如果两个时间相等，就是获取当前时间的能耗价格"""

        if duration_start_time == 0:  # 意味着这个机器没有启用
            return 0

        # 将时间转化为datetime.datetime类型
        duration_start_time = self.toDatetime(duration_start_time)
        duration_end_time = self.toDatetime(duration_end_time)

        start_date = duration_start_time.date()
        end_date = duration_end_time.date()

        if duration_start_time == duration_end_time:  # 是一个时间点
            date_factor = self.price.getDateFactor(start_date)
            duration_start_time = duration_start_time.strftime("%H:%M")
            if duration_start_time in self.day_shift:
                return self.price.energy_price_day * date_factor
            elif duration_start_time in self.evening_shift:
                return self.price.energy_price_evening * date_factor
            else:
                return self.price.energy_price_night * date_factor
        elif start_date == end_date:  # 该时间段在一天内
            duration_start_time = duration_start_time.strftime("%H:%M")
            duration_end_time = duration_end_time.strftime("%H:%M")
            duration = IntervalSet.between(duration_start_time, duration_end_time)  # 当前时间段
            date_factor = self.price.getDateFactor(start_date)
            day_hours = self.getOverlapHours(duration, self.day_shift)
            evening_hours = self.getOverlapHours(duration, self.evening_shift)
            night_hours = self.getOverlapHours(duration, self.night_shift0) + self.getOverlapHours(duration,
                                                                                                   self.night_shift1)
            price = day_hours * self.price.energy_price_day + evening_hours * self.price.energy_price_evening + night_hours * self.price.energy_price_night
            total_price = price * date_factor
            return total_price
        else:  # 该时间段跨了多天
            duration_start_time = duration_start_time.strftime("%H:%M")
            duration_end_time = duration_end_time.strftime("%H:%M")
            date = start_date
            total_price = 0
            while date <= end_date:
                date_factor = self.price.getDateFactor(date)
                if date == start_date:
                    duration = IntervalSet.between(duration_start_time, "23:59")
                elif date == end_date:
                    duration = IntervalSet.between("00:00", duration_end_time)
                else:
                    duration = IntervalSet.between("00:00", "23:59")
                day_hours = self.getOverlapHours(duration, self.day_shift)
                evening_hours = self.getOverlapHours(duration, self.evening_shift)
                night_hours = self.getOverlapHours(duration, self.night_shift0) + self.getOverlapHours(duration,
                                                                                                       self.night_shift1)
                total_price += (
                                       day_hours * self.price.energy_price_day + evening_hours * self.price.energy_price_evening
                                       + night_hours * self.price.energy_price_night) * date_factor
                date += datetime.timedelta(days=1)
            return total_price

    def getEnergyCostFromSchedule(self, schedule, machine_first_start_time):
        """基于schedule计算能耗成本"""

        def getEnergyPrice(duration_start_time, duration_end_time):
            """获取当前时间段下的能耗价格*时长；如果两个时间相等，就是获取当前时间的能耗价格"""

            if duration_start_time == 0:  # 意味着这个机器没有启用
                return 0

            try:
                duration_start_time = duration_start_time.astype(datetime.datetime)
                duration_end_time = duration_end_time.astype(datetime.datetime)
            except AttributeError:
                duration_start_time = datetime.datetime.strptime(duration_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                                                                 '%Y-%m-%d %H:%M:%S')
                duration_end_time = datetime.datetime.strptime(duration_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                                                               '%Y-%m-%d %H:%M:%S')
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
                day_hours = self.getOverlapHours(duration, self.day_shift)
                evening_hours = self.getOverlapHours(duration, self.evening_shift)
                night_hours = self.getOverlapHours(duration, self.night_shift0) + self.getOverlapHours(duration,
                                                                                                       self.night_shift1)
                price = day_hours * self.price.energy_price_day + evening_hours * self.price.energy_price_evening + night_hours * self.price.energy_price_night
                total_price = price * date_factor
                return total_price
            else:
                date_factor0 = self.price.getDateFactor(start_date)  # 第一天
                date_factor1 = self.price.getDateFactor(end_date)  # 第二天
                duration0 = IntervalSet.between(duration_start_time, "23:59")
                duration1 = IntervalSet.between("00:00", duration_end_time)
                '''由于一个duration（生产或空转）不会超过三小时，为了减少计算时间，只计算以下班次的重叠部分'''
                evening_hours = self.getOverlapHours(duration0, self.evening_shift)
                night_hours0 = self.getOverlapHours(duration0, self.night_shift0)
                night_hours1 = self.getOverlapHours(duration1, self.night_shift1)  # 第二天夜班的重叠时长
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
            energy_consumption = self.getMachineEnergyConsume(machine_id, machine_status)
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

    def getEnergyCost(self, schedule_for_cal, machine_first_start_time):
        """基于schedule_for_cal计算能耗成本"""
        total_energy_cost = 0
        machine_first_start_price = np.zeros(shape=self.DATA.machine_num)  # 机器第一次开机时的能耗价格

        for _, row in schedule_for_cal.iterrows():
            machine_id = row['Machine']
            machine_status = row['Machine Status']
            start_time = row['Start Time']
            end_time = row['End Time']
            energy_consumption = self.getMachineEnergyConsume(machine_id, machine_status)
            if machine_status == "停机":
                energy_price = self.getEnergyPrice(end_time, end_time)
            else:
                energy_price = self.getEnergyPrice(start_time, end_time)
            total_energy_cost += energy_consumption * energy_price

        # 计算机器第一次开机时的能耗成本
        for i, first_start_time in enumerate(machine_first_start_time):
            machine_first_start_price[i] = self.getEnergyPrice(first_start_time, first_start_time)
        first_start_energy_cost = self.DATA.startup_energy_consumption * machine_first_start_price

        total_energy_cost += first_start_energy_cost.sum()
        return total_energy_cost

    def getLaborCost(self, schedule_for_cal):
        """计算当前调度表下的人工成本"""
        schedule = schedule_for_cal[
            (schedule_for_cal['Machine Status'] == '生产') | (schedule_for_cal['Machine Status'] == '空转')]
        machine_list = schedule.Machine.unique()
        total_labor_cost = 0
        for machine_id in machine_list:  # 遍历所有用到的machine
            shift_hours = {}
            procedure_id = 0
            for _, row in schedule[schedule['Machine'] == machine_id].iterrows():
                start_time = row['Start Time']
                end_time = row['End Time']
                procedure_id = row['Procedure']
                shift_hours = self.getLaborHours(start_time, end_time, shift_hours)
            labor_cost_factor = self.DATA.cost_for_procedure.loc[procedure_id, '人工成本因子']
            for shifts in shift_hours.values():  # 遍历该机器在某一天的加工时间
                for shift in shifts.keys():  # 遍历该天的三个班段
                    if 0 < shifts[shift] <= 4:
                        shifts[shift] = 4
                    elif shifts[shift] > 4:
                        shifts[shift] = 8
                    if shift == 'day':
                        total_labor_cost += shifts[shift] * self.price.labor_price_day * labor_cost_factor
                    elif shift == 'evening':
                        total_labor_cost += shifts[shift] * self.price.labor_price_evening * labor_cost_factor
                    else:
                        total_labor_cost += shifts[shift] * self.price.labor_price_night * labor_cost_factor
        return total_labor_cost

    def getHoldCost(self, schedule_item):
        """计算当前调度表下的库存成本"""
        schedule_item = schedule_item[schedule_item['Machine Status'] == '生产']
        total_hold_cost = 0
        for category_id, category_group in schedule_item.groupby('Category ID'):  # 遍历某一类产品
            procedure_end_time = 0
            total_hold_time_of_category = []  # 该类别下所有产品的总库存时间
            for procedure, procedure_group in category_group.groupby('Procedure'):  # 遍历该类产品的各道工序
                procedure_start_time = procedure_group['Start Time'].map(self.toDatetime)
                if isinstance(procedure_end_time, pd.Series):
                    procedure_start_time = procedure_start_time.reset_index(drop=True)
                    procedure_end_time = procedure_end_time.reset_index(drop=True)
                    hold_time_before_this_procedure = procedure_start_time - procedure_end_time  # 计算各工序前的产品滞留时间
                    hold_time_before_this_procedure = np.array(
                        hold_time_before_this_procedure.map(lambda x: int(x.days)))  # 每满24h，算一天的库存成本
                    if len(total_hold_time_of_category) == 0:
                        total_hold_time_of_category = hold_time_before_this_procedure
                    else:
                        total_hold_time_of_category += hold_time_before_this_procedure
                procedure_end_time = procedure_group['End Time'].map(self.toDatetime)

            procedure_end_time = procedure_end_time.reset_index(drop=True)
            hold_time_before_delivery = procedure_end_time.map(lambda x: procedure_end_time.max() - x)  # 产品完工后到发货前的库存时间
            hold_time_before_delivery = np.array(hold_time_before_delivery.map(lambda x: int(x.days)))
            total_hold_time_of_category += hold_time_before_delivery
            total_hold_time_of_category = total_hold_time_of_category.sum()
            total_hold_cost += total_hold_time_of_category * self.DATA.cost_factor_for_category.loc[
                category_id, '仓储成本系数'] * self.price.hold_price  # 累加上该类别所有产品的库存成本

        return total_hold_cost

    def getDelayCost(self, schedule_order):
        """计算当前调度表下的延迟成本"""
        schedule_order = schedule_order[schedule_order['Machine Status'] == '生产']
        order_end_time = schedule_order.groupby("Order ID")[['End Time', 'Category ID']].max()  # 每个订单的完工时间
        order_num = self.DATA.order_table["数量"]
        order_due_time = self.DATA.order_table["最迟交期"]
        order_delay_time = order_end_time['End Time'] - order_due_time
        order_delay_days = order_delay_time.map(lambda x: 0 if int(x.days) < 0 else int(x.days))
        order_hold_cost_factor = order_end_time['Category ID'].map(
            lambda x: self.DATA.cost_factor_for_category.loc[x, '订单延迟成本系数'])  # 每个产品对应类别的延迟成本系数

        order_delay_days = np.array(order_delay_days)
        order_num = np.array(order_num)
        order_hold_cost_factor = np.array(order_hold_cost_factor)

        total_delay_cost = (order_delay_days * order_num) @ order_hold_cost_factor * self.price.delay_price

        return total_delay_cost

    @staticmethod
    def time_to_str(t):
        """把python的时间（如datetime64、Timestamp）转化为字符串"""
        if isinstance(t, np.datetime64):
            t = t.astype(datetime.datetime)
        try:
            return t.strftime('%Y%m%d%H%M%S')
        except ValueError:
            print(ValueError)

    @staticmethod
    def toDatetime(time):
        """将时间转化为datetime.datetime类型"""
        try:
            new_time = time.astype(datetime.datetime)
        except AttributeError:
            new_time = datetime.datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        return new_time


class StrategyData:
    """策略1：同类订单一起加工的数据准备"""

    def __init__(self, job_strategy, machine_strategy, time_strategy, nwn_rate=0, wn_rate=0):
        self.DATA = DATA
        self.machine_information = pd.concat([DATA.machine_information, DATA.machine_energy_consumption], axis=1)
        self.nwn_rate = nwn_rate  # no work at night rate
        self.wn_rate = wn_rate  # work at night rate

        """job"""
        self.category_id_list = []
        self.model_id_list = []
        if job_strategy == 'category_together':
            self.category_id_list = list(set(DATA.order_table["产品类别"]))
            self.job_id_list = self.category_id_list
        elif job_strategy == 'model_together':
            self.model_id_list = list(set(DATA.order_table["产品型号"]))
            self.job_id_list = self.model_id_list
        self.job_num = len(self.job_id_list)
        self.jobs = self.getJobs(self.category_id_list, self.model_id_list)

        """machine"""
        if machine_strategy == 'machine_selection':
            self.procedure_id_list = DATA.machine_num_of_procedure.index.tolist()
            self.machine_id_list = self.machineSelection()
            print(f'采用设备选型策略，选择的设备为：{self.machine_id_list}。')
        else:
            self.machine_id_list = DATA.machine_list
        self.machine_num = len(self.machine_id_list)
        self.machines = self.getMachines()

        """time"""
        if time_strategy == 'no_work_at_night':
            self.setNoWorkAtNight()
            self.setWorkAtNight()

    def machineSelection(self):
        """按启发式规则生成初始解"""
        # 保留每道工序的加工机器中能耗最小的机器
        machine_id_list = []
        mean_energy_cost = self.machine_information.groupby("对应工序")[
            ['生产能耗/小时', '空转(开机等待)能耗/小时', '开机一次性能耗']].mean()
        for procedure_id in self.procedure_id_list:
            mean_energy_consume_produce = mean_energy_cost.loc[procedure_id, '生产能耗/小时']
            mean_energy_consume_idle = mean_energy_cost.loc[procedure_id, '空转(开机等待)能耗/小时']
            mean_energy_consume_startup = mean_energy_cost.loc[procedure_id, '开机一次性能耗']
            keep_machine = list(self.machine_information[
                                    (self.machine_information["对应工序"] == procedure_id) &
                                    (self.machine_information["生产能耗/小时"] <= mean_energy_consume_produce) &
                                    (self.machine_information['空转(开机等待)能耗/小时'] <= mean_energy_consume_idle) &
                                    (self.machine_information['开机一次性能耗'] <= mean_energy_consume_startup)
                                    ].index)
            machine_id_list.extend(keep_machine)
        for job in self.jobs:  # 遍历所有的job
            for procedure_machines in job.machines:  # 遍历该job的所有工序
                if not set(procedure_machines).intersection(machine_id_list):  # 如果选择的设备不足以加工该job的此道工序
                    produce_energy_cost = self.machine_information.loc[procedure_machines, '生产能耗/小时']
                    choose_machine = produce_energy_cost.idxmin()
                    machine_id_list.append(choose_machine)

        return machine_id_list

    def setNoWorkAtNight(self):
        """设置不能在晚上(00:00-08:00)生产的设备"""
        machine_information = self.machine_information.loc[self.machine_id_list, :].sort_values(
            by='生产能耗/小时')  # 按生产能耗从小到大排序
        no_work_at_night_machine_ids = machine_information.iloc[:int(self.machine_num * self.nwn_rate), :].index  # 取生产能耗最小的前int(self.machine_num*0.1)台设备
        print(f'不能在晚上(00:00-08:00)生产的设备有：{no_work_at_night_machine_ids}')
        for machine_id in no_work_at_night_machine_ids:
            self.machines.loc[machine_id].no_work_at_night = 1

    def setWorkAtNight(self):
        """设置只能在晚上(22:00-06:00)生产的设备"""
        machine_information = self.machine_information.loc[self.machine_id_list, :].sort_values(by='生产能耗/小时',
                                                                                                ascending=False)  # 按生产能耗从大到小排序
        work_at_night_machine_ids = machine_information.iloc[:int(self.machine_num * self.wn_rate), :].index  # 取生产能耗最大的前int(self.machine_num*0.1)台设备
        print(f'只能在晚上(22:00-06:00)生产的设备有：{work_at_night_machine_ids}')
        for machine_id in work_at_night_machine_ids:
            self.machines.loc[machine_id].work_at_night = 1

    @staticmethod
    def getJobs(category_list, model_list):
        jobs = []
        if category_list:
            for category_id in category_list:
                job = Job(category_id=category_id)
                jobs.append(job)
            jobs = pd.Series(jobs)
            jobs.index = category_list
        else:
            for model_id in model_list:
                job = Job(model_id=model_id)
                jobs.append(job)
            jobs = pd.Series(jobs)
            jobs.index = model_list
        return jobs

    def getMachines(self):
        """获取所有的机器对象"""
        machines = []
        for machine_id in self.machine_id_list:
            machine = Machine(machine_id)
            machines.append(machine)
        machines = pd.Series(machines)
        machines.index = self.machine_id_list
        return machines


class GAData:
    def __init__(self, strategy_data):
        self.job_num = strategy_data.job_num
        self.job_id_list = strategy_data.job_id_list
        self.jobs = strategy_data.jobs

        self.machine_num = strategy_data.machine_num
        self.machine_id_list = strategy_data.machine_id_list
        self.machines = strategy_data.machines

        self.chromosome_size = sum(map(lambda x: x.procedure_num, self.jobs))

    @staticmethod
    def resetMachine(machines):
        """每遍历一个个体，重置一次，否则这些属性值会进入下一次迭代"""
        for machine in machines:
            machine.reset()

    @staticmethod
    def resetJob(jobs):
        for job in jobs:
            job.reset()


def getStrategyData(file_name, order_name, order_time, job_strategy='', machine_strategy='', time_strategy='', nwn_rate=0, wn_rate=0):
    origin_data = Data(file_name, order_name, order_time)  # 对全局变量origin_data进行修改
    importGlobalData(origin_data)  # 把origin_data导入datav20221110模块，一次性的
    strategy_data = StrategyData(job_strategy, machine_strategy, time_strategy, nwn_rate, wn_rate)
    return strategy_data
