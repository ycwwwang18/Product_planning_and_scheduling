import re
import calendar
import datetime
import numpy as np
import pandas as pd


class Data:
    """数据类"""

    def __init__(self, file_path, order_name, order_date):
        data = pd.read_excel(file_path, sheet_name=None)

        '''产品相关信息'''
        # 1.订单相关信息-对应型号
        self.order_name = order_name
        self.order_time = np.datetime64(order_date)
        self.order1 = data["订单信息_1"]
        self.order2 = data["订单信息_2"]

        # 2.类别相关信息
        self.category_to_model = data["产品信息"]
        # 2.1.工序相关信息
        self.procedure_for_category = data["产品类别工序"]
        # 2.2.成本相关信息
        self.cost_factor_for_category = data["产品类别"]

        '''机器相关信息'''
        # 1.机器类型相关信息-对应工序和产品类别
        self.machine_information = data["设备信息"]
        # 2.时间相关信息
        self.machine_maintenance = data["设备维保信息"]
        # 3.成本相关信息
        self.energy_price = data["能耗价格(PH)"]
        self.machine_energy_consumption = data["设备稼动能耗信息"]
        self.startup_energy_consumption = self.machine_energy_consumption['开机一次性能耗'].values

        '''工序相关信息'''
        # 1.时间相关信息
        self.model_change_time = data["工序准备时间"]
        self.processing_time = data["设备作业时间"]  # 对应产品类别
        # 2.成本相关信息
        self.cost_for_procedure = data["工序作业成本"]

        '''其他信息'''
        self.other_information = data["其他信息"]
        self.lead_time = self.getLeadTime()

        '''数据格式处理'''
        self.dataReforming()
        self.order_table = self.orderRangeSelect()  # 选择要进行排程的订单范围

        '''机器的维修日期'''
        self.machine_num = self.getMachineNum()
        self.machine_list = self.getMachine()
        self.machine_information = self.machineInformationReforming()

        '''输出参数'''
        self.category_list = list(self.procedure_for_category.index)  # 所有产品类别
        self.category_num = len(self.category_list)  # 产品类别数目
        self.procedure_of_category = self.getProcedureOfCategory()  # 所有产品类别对应的工序
        self.procedure_num_of_category = self.getProcedureNumOfCategory()  # 所有产品类别的工序道数
        self.machine_of_category = self.getMachineOfCategory()  # 所有产品类别的每道工序的加工机器列表
        self.order_earliest_start_time = self.order_time + self.lead_time

        # self.orders = self.getOrders()  # 所有的订单

        def getFeasibleCategory(category_string):
            """获取某一机器对应的可选产品类别集合，返回list"""
            category_list = category_string.split(",")
            category_list = list(map(lambda y: int(y[1:]), category_list))
            return category_list

        self.machine_information["对应产品类别"] = self.machine_information.对应产品类别.map(getFeasibleCategory)

    '''辅助计算和处理'''

    def orderRangeSelect(self):
        """选择指定日期范围的某个订单数据，返回新订单"""
        new_order = None
        if self.order_name == "order1":
            new_order = self.order1[self.order1["订单日期"] == self.order_time]
        elif self.order_name == "order2":
            new_order = self.order2[self.order2["订单日期"] == self.order_time]
        category = []
        for index, row in new_order.iterrows():  # 遍历订单数据的所有行
            category.append(self.category_to_model.loc[row["产品型号"], "产品类别"])
        new_order.insert(4, "产品类别", category)
        return new_order

    def dataReforming(self):
        """删掉不需要的行或列，修改格式"""
        # 订单1
        order1_columns = self.order1.columns
        self.order1.drop(columns=order1_columns[9:], inplace=True)
        self.order1["产品型号"] = self.order1.产品型号.map(lambda x: int(x[1:]))
        self.order1.set_index("订单编号", inplace=True)

        # 订单2
        order2_columns = self.order2.columns
        self.order2.drop(columns=order2_columns[9:], inplace=True)
        self.order2["产品型号"] = self.order2.产品型号.map(lambda x: int(x[1:]))
        self.order2.set_index("订单编号", inplace=True)

        # 产品类别-库存和延迟成本系数
        cost_factor_for_category_columns = self.cost_factor_for_category.columns
        self.cost_factor_for_category.drop(columns=cost_factor_for_category_columns[1:3], inplace=True)
        self.cost_factor_for_category = self.cost_factor_for_category.iloc[:, :3]
        self.cost_factor_for_category["产品类别"] = self.cost_factor_for_category.产品类别.map(lambda x: int(x[1:]))
        self.cost_factor_for_category.set_index("产品类别", inplace=True)

        # 产品型号对应的类别
        self.category_to_model.drop(columns=["产品名称"], inplace=True)
        self.category_to_model["产品型号"] = self.category_to_model.产品型号.map(lambda x: int(x[1:]))
        self.category_to_model["产品类别"] = self.category_to_model.产品类别.map(lambda x: int(x[1:]))
        self.category_to_model.set_index("产品型号", inplace=True)

        # 产品类别工序
        self.procedure_for_category["类别序号"] = self.procedure_for_category.类别序号.map(lambda x: int(x[1:]))
        self.procedure_for_category.set_index("类别序号", inplace=True)
        self.procedure_for_category.columns = list(map(lambda x: int(x[2:]), self.procedure_for_category.columns))

        # 设备信息
        machine_information_columns = self.machine_information.columns
        self.machine_information.drop(columns=machine_information_columns[3:5], inplace=True)
        self.machine_information["设备编号"] = self.machine_information.设备编号.map(lambda x: int(x[1:]))
        self.machine_information["对应工序"] = self.machine_information.对应工序.map(lambda x: int(x[1:]))
        self.machine_information.set_index("设备编号", inplace=True)

        # 工序作业成本
        self.cost_for_procedure.columns = list(self.cost_for_procedure.iloc[0, :])
        self.cost_for_procedure.drop(axis=0, index=0, inplace=True)
        self.cost_for_procedure["工序编号"] = self.cost_for_procedure.工序编号.map(lambda x: int(x[1:]))
        self.cost_for_procedure.set_index("工序编号", inplace=True)
        self.cost_for_procedure.columns = ['夜班', '白班', '晚班', '人工成本因子', '计件成本因子']

        # 设备作业时间
        self.processing_time["设备编号"] = self.processing_time.设备编号.map(lambda x: int(x[1:]))
        self.processing_time.set_index("设备编号", inplace=True)
        self.processing_time.columns = list(map(lambda x: int(x[4:]), self.processing_time.columns))

        def time_format(x):
            """将元素转化为时间格式"""
            if np.isnan(x):
                return None
            else:
                return np.timedelta64(int(x), 'm')

        self.processing_time = self.processing_time.applymap(time_format)

        # 设备维保信息
        self.machine_maintenance["设备编号"] = self.machine_maintenance.设备编号.map(lambda x: int(x[1:]))
        self.machine_maintenance.set_index("设备编号", inplace=True)

        # 设备能耗信息
        self.machine_energy_consumption["设备编号"] = self.machine_energy_consumption.设备编号.map(lambda x: int(x[1:]))
        self.machine_energy_consumption.set_index("设备编号", inplace=True)
        self.machine_energy_consumption["生产间隔时间超过则停机小时数"] = \
            self.machine_energy_consumption.生产间隔时间超过则停机小时数.map(lambda x: np.timedelta64(int(x * 60), 'm'))

        # 能耗价格
        self.energy_price = self.energy_price.iloc[:, :5]
        self.energy_price.columns = ['日期', '夜班', '白班', '晚班', '日期价格因子']
        self.energy_price["日期"] = self.energy_price.日期.map(lambda x: x.date())

        # 换型准备时间
        self.model_change_time["工序编号"] = self.model_change_time.工序编号.map(lambda x: int(x[1:]))
        self.model_change_time["换型准备时间"] = self.model_change_time.换型准备时间.map(
            lambda x: np.timedelta64(int(x), 'm'))
        self.model_change_time.set_index("工序编号", inplace=True)

    def machineInformationReforming(self):
        """修改self.machine_information的格式，将换型时间和维修期合并进去"""
        maintenance_day = self.getMaintenanceDay()
        machine_change_model_time = []
        machine_information = self.machine_information
        for machine_id in machine_information.index:
            procedure_id = machine_information.loc[machine_id, "对应工序"]
            procedure_change_model_time = self.model_change_time.loc[procedure_id, "换型准备时间"]
            machine_change_model_time.append(procedure_change_model_time)
        machine_information.insert(3, "换型时间", machine_change_model_time)
        machine_information.insert(4, "维保日期", maintenance_day)

        return machine_information

    def getMachine(self):
        """获取所有机器集合，返回list"""
        machine_list = list(self.machine_information.index)
        return machine_list

    def getLeadTime(self):
        """获取订单加工的提前期"""
        word = self.other_information.iloc[11, 0]
        lead_time = int(re.findall("\\d+", word)[1])
        return np.timedelta64(lead_time, 'h')

    def getMachineNum(self):
        """获取所有机器个数，返回int"""
        return self.machine_information.shape[0]

    def getProcedureNumOfCategory(self):
        """获取所有产品类别对应的工序数量，返回list"""
        procedure_num_list = []
        for category in self.category_list:  # 遍历所有的产品类别
            procedure_array = self.procedure_for_category.loc[category].to_numpy()
            procedure_num_of_category = procedure_array.sum()
            procedure_num_list.append(procedure_num_of_category)
        return procedure_num_list

    def getMachineOfCategory(self):
        """获取所有产品类别各道工序的对应加工机器，返回list"""

        def get_feasible_machine_for_category(category_num, procedure_num):
            """获取某一产品类别的某道工序对应的可选机器集合，返回list"""
            feasible_machine = self.machine_information[
                (self.machine_information["对应工序"] == procedure_num) &
                (self.machine_information["对应产品类别"].str.contains(str(category_num)))
                ].index
            return list(feasible_machine)

        machine_list = []
        for category in self.category_list:
            category_machine_list = []
            for procedure in self.procedure_of_category[category - 1]:
                feasible_machine_list = get_feasible_machine_for_category(category, procedure)
                category_machine_list.append(feasible_machine_list)
            machine_list.append(category_machine_list)
        return machine_list

    def getProcedureOfCategory(self):
        """获取所有产品类别对应的工序集合，返回list"""
        procedure_of_category_list = []
        for x in self.category_list:
            procedure_list = []
            for y in range(1, self.procedure_for_category.shape[1] + 1):  # 遍历所有工序序号
                if self.procedure_for_category.loc[x, y] == 1:
                    procedure_list.append(y)
            procedure_of_category_list.append(procedure_list)
        return procedure_of_category_list

    def getMaintenanceDay(self):
        """获取所有机器在下单日期所在月和下个月的维保期，返回list[datetime64]"""

        def getMachineMaintenanceDayOfMonth(machine_id, date):
            """获得某机器在某日期所在月的维保期，返回datetime"""

            def get_last_day_of_month(query_date):
                """获取某日期所在月的最后一天的日期，返回datetime"""
                year = query_date.year
                month = query_date.month
                last_day = calendar.monthrange(year, month)[1]
                last_day = datetime.datetime(year, month, last_day)
                return last_day

            maintenance_date = self.machine_maintenance.loc[machine_id, "停机维保日期信息"]
            week_name_of_maintenance_date = maintenance_date[-2:]  # 提取周几
            week_num_of_maintenance_date = maintenance_date[2:-2]  # 提取第几个周

            first_day_of_month = datetime.datetime(date.year, date.month, 1)
            last_day_of_month = get_last_day_of_month(date)

            if week_name_of_maintenance_date == "周六":
                query_day_week_name_num = 5
            else:
                query_day_week_name_num = 6

            if week_num_of_maintenance_date == "最后一个":
                last_day_week_name_num = last_day_of_month.weekday()
                week_delta = query_day_week_name_num - last_day_week_name_num
                if week_delta > 0:
                    maintenance_date_of_month = last_day_of_month - datetime.timedelta(days=7 - week_delta)
                else:
                    maintenance_date_of_month = last_day_of_month + datetime.timedelta(days=week_delta)
            else:  # 第二个
                first_day_week_name_num = first_day_of_month.weekday()
                week_delta = query_day_week_name_num - first_day_week_name_num
                if week_delta < 0:
                    maintenance_date_of_month = first_day_of_month + datetime.timedelta(days=14 + week_delta)
                else:
                    maintenance_date_of_month = first_day_of_month + datetime.timedelta(days=7 + week_delta)

            return maintenance_date_of_month

        maintenance_day_list = []
        order_time_dt = self.order_time.astype(datetime.datetime)  # 转化下单时间的格式为datetime
        for m in self.machine_list:  # 遍历所有的机器
            # 找到下单日期所在月的下个月
            try:
                order_time_next_month = order_time_dt.replace(month=order_time_dt.month + 1, day=1)
            except ValueError:
                if order_time_dt.month == 12:
                    order_time_next_month = order_time_dt.replace(year=order_time_dt.year + 1, month=1, day=1)
                else:
                    raise
            maintenance_day = [np.datetime64(getMachineMaintenanceDayOfMonth(m, order_time_dt)),
                               np.datetime64(getMachineMaintenanceDayOfMonth(m, order_time_next_month))]
            maintenance_day_list.append(maintenance_day)
        return maintenance_day_list