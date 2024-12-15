import os
import random
from datetime import datetime, timedelta
import math
from OrderIntake.OrderManager import OrderIntake
from Simulation.Simul8 import Simul8API
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np

def sim_time_to_date(value):
    """
    Converts simulation time (windows datetime) to date.

    Parameters
    ----------
        value: float
            The simulation time in windows time format

    :return:
        The simulation time as a date

    """
    if not type(value) == float:
        value = float(value)

    windows_start_date = datetime(1900, 1, 1)

    end_date = windows_start_date + timedelta(days=value)

    return end_date

def time_remaining(order: pd.DataFrame, processing_times: dict):
    """
    Calculates the remaining processing time of an order.

    Parameters
    ----------
        order: pd.DataFrame
            The order for which the remaining processing time is to be calculated
        processing_time: dict
            A dict that contains the process times assigned to the stations that can be used to process an order

    :return:
        The remaining processing time to complete the order

    """
    pt_s = processing_times["S"]
    pt_m = processing_times["M"]
    pt_mf = processing_times["MF"]
    pt_e = processing_times["E"]
    pt_eh = processing_times["EH"]
    pt_vb = processing_times["VB"]
    pt_swd = processing_times["SWD"]
    pt_vsw = processing_times["VSW"]
    pt_fd = processing_times["FD"]
    pt_vfd = processing_times["VFD"]

    order_specific_process_times = []
    t_s = order.at[0, "S"] * pt_s
    order_specific_process_times.append(t_s)
    t_m = order.at[0, "M"] * pt_m
    order_specific_process_times.append(t_m)
    t_mf = order.at[0, "MF"] * pt_mf
    order_specific_process_times.append(t_mf)
    t_e = order.at[0, "E"] * pt_e
    order_specific_process_times.append(t_e)
    t_eh = order.at[0, "EH"] * pt_eh
    order_specific_process_times.append(t_eh)
    t_vb = order.at[0, "VB"] * pt_vb
    order_specific_process_times.append(t_vb)
    t_swd = order.at[0, "SWD"] * pt_swd
    order_specific_process_times.append(t_swd)
    t_vsw = order.at[0, "VSW"] * pt_vsw
    order_specific_process_times.append(t_vsw)
    t_fd = order.at[0, "FD"] * pt_fd
    order_specific_process_times.append(t_fd)
    t_vfd = order.at[0, "VFD"] * pt_vfd
    order_specific_process_times.append(t_vfd)

    min_process_time_to_complete_one_component = sum(order_specific_process_times)
    longest_process_time_on_one_machine = max(order_specific_process_times)

    finished_components = 0
    t_remaining = 0
    for i in range(len(order)):
        t_remaining = 0
        t_remaining += order.at[i, "S"] * abs(order.at[i, "S_Done"] - 1) * pt_s
        t_remaining += order.at[i, "M"] * abs(order.at[i, "M_Done"] - 1) * pt_m
        t_remaining += order.at[i, "MF"] * abs(order.at[i, "MF_Done"] - 1) * pt_mf
        t_remaining += order.at[i, "E"] * abs(order.at[i, "E_Done"] - 1) * pt_e
        t_remaining += order.at[i, "EH"] * abs(order.at[i, "EH_Done"] - 1) * pt_eh
        t_remaining += order.at[i, "VB"] * abs(order.at[i, "VB_Done"] - 1) * pt_vb
        t_remaining += order.at[i, "SWD"] * abs(order.at[i, "SWD_Done"] - 1) * pt_swd
        t_remaining += order.at[i, "VSW"] * abs(order.at[i, "VSW_Done"] - 1) * pt_vsw
        t_remaining += order.at[i, "FD"] * abs(order.at[i, "FD_Done"] - 1) * pt_fd
        t_remaining += order.at[i, "VFD"] * abs(order.at[i, "VFD_Done"] - 1) * pt_vfd
        if t_remaining == 0:
            finished_components += 1
        else:
            break

    if finished_components == 0:
        process_time_so_far = min_process_time_to_complete_one_component - t_remaining
    else:
        process_time_so_far_of_last_completed_component = min_process_time_to_complete_one_component + (
                finished_components - 1) * longest_process_time_on_one_machine
        process_time_so_far_of_last_uncompleted_component = min_process_time_to_complete_one_component - t_remaining + finished_components * longest_process_time_on_one_machine
        if process_time_so_far_of_last_completed_component == process_time_so_far_of_last_uncompleted_component:
            process_time_so_far = process_time_so_far_of_last_completed_component
        else:
            process_time_so_far = process_time_so_far_of_last_completed_component + (
                    process_time_so_far_of_last_uncompleted_component - process_time_so_far_of_last_completed_component)

    total_processing_time_of_order = min_process_time_to_complete_one_component + (
            len(order) - 1) * longest_process_time_on_one_machine
    total_time_remaining = total_processing_time_of_order - process_time_so_far

    return round(((total_time_remaining / 3600) / 24), 3)

def waiting_time(order: pd.DataFrame, t_step: float):
    """
    Calculates the time a job has spent in the current queue.

    Parameters
    ----------
        order: pd.DataFrame
            The order for which the remaining processing time is to be calculated
        t_step: float
            The simulation time at the current step

    :return:
        The time that the submitted order has spent in the current queue

    """
    wt = t_step - (float(order.at[0, "t_slot"]))
    return round(wt, 3)

def time_until_due_date(order: pd.DataFrame, t_step: float):
    """
    Calculates the time until the due date of a job.

    Parameters
    ----------
        order: pd.DataFrame
            The order for which the remaining processing time is to be calculated
        t_step: float
            The simulation time at the current step

    :return:
        The time until the due date of the submitted job

    """
    ttd = (float(order.at[0, "Deadline"])) - t_step
    return round(ttd, 3)

def slack_time(order: pd.DataFrame, processing_times: dict, t_step: float):
    """
    Calculates the slack time of a job.

    Parameters
    ----------
        order: pd.DataFrame
            The order for which the remaining processing time is to be calculated
        processing_time: dict
            A dict that contains the process times assigned to the stations that can be used to process an order

    :return:
        The slack time of the submitted job

    """
    tr = time_remaining(order, processing_times)
    st = (float(order.at[0, "Deadline"])) - (t_step + tr)
    return round(st, 3)

def machine_utilization(order: pd.DataFrame, t_step: float):
    """
    Calculates the machine utilization of the machine a submitted job is processed on.

    Parameters
    ----------
        order: pd.DataFrame
            The order for which the remaining processing time is to be calculated
        t_step: float
            The simulation time at the current step

    :return:
        The machine utilization of the machine the submitted job is processed on.

    """
    t_start = order.at[0, "t_start"]
    t_start_waiting = order.at[0, "t_start_waiting"]
    t_stop_waiting = order.at[0, "t_stop_waiting"]

    if not t_start_waiting == 0 and not t_stop_waiting == 0:
        utilization = round(1 - ((t_step - t_start_waiting) - (t_step - t_stop_waiting)) / (t_step - t_start) , 3)
        return utilization

    elif not t_start_waiting == 0:
        utilization = round(1 - (t_step - t_start_waiting) / (t_step - t_start), 3)
        return utilization

    return 1.0


class BaseTemplate:
    """
        A base class that contains an order book and basic functions for processing this order book.

        ...

        Attributes
        ----------
        problem : Simul8Problem
            An instance of the Simul8 problem class
        columns: list
            A list containing the column names of the order book (DataFrame)
        df : int
            A DataFrame that is used as an order book to store general information about orders

        Methods
        -------
        len()
            Returns the length of the order book
        duplicate(item)
            Checks whether an item is already contained in the order book
        append(element)
            Appends a passed element to the order book
        remove_order(order_number)
            Removes all items holding the assigned order number
        get_order(order_number)
            Returns the order holding the assigned order number
        get_first_order()
            Returns the first order contained in the order book
        get_included_order()
            Returns the first items of all orders contained in the order book
        get_random_order()
            Returns a random order contained in the order book
        is_empty()
            Returns True if the order book is empty and False if not
        clear()
            Clears the order intake data frame
        """

    def __init__(self, problem):

        self.problem = problem
        self.columns = ["OrderNumber", "ItemNumber", "ItemType", "Deadline", "t_slot",
                        "S", "S_Done", "M", "M_Done", "MF", "MF_Done", "E", "E_Done", "EH", "EH_Done", "VB", "VB_Done",
                        "SWD", "SWD_Done", "VSW", "VSW_Done", "FD", "FD_Done", "VFD", "VFD_Done", "InQueue", "t_start", "t_start_waiting"]
        self.df = pd.DataFrame(columns=self.columns)

    def __len__(self) -> len:
        """
        Returns the length of the order book.

        :return:
            The length of the order book.

        """
        if not self.df.empty:
            included_orders = self.df.drop_duplicates(subset=["OrderNumber"])
            order_numbers = included_orders["OrderNumber"].to_list()
            return len(order_numbers)
        else:
            return 0

    def duplicate(self, item: pd.DataFrame) -> bool:
        """
        Checks whether an item is already contained in the order book.

        Parameters
        ----------
            item: pd.DataFrame
                The order for which it is checked whether its order number is already contained in the order book

        :return:
            True if the order number is already contained in the order book and False if not

        """
        if item["OrderNumber"] in self.df["OrderNumber"]:
            return True
        else:
            return False

    def append(self, element: pd.DataFrame) -> None:
        """
        Appends a passed element to the order book.

        Parameters
        ----------
            element: pd.DataFrame
                The element to append to the order book

        """
        self.df = pd.concat([self.df, element], ignore_index=True)

    def remove_order(self, order_number: int | str) -> None:
        """
        Removes all items holding the assigned order number.

        Parameters
        ----------
            order_number: int | str
                The order number of the order to be removed

        """
        mask = self.df["OrderNumber"] == int(order_number)
        self.df = self.df[~mask]

    def get_order(self, order_number: int | str) -> pd.DataFrame:
        """
        Returns the order holding the assigned order number.

        Parameters
        ----------
            order_number: int | str
                The order number of the order to be returned

        :return:
            The order holding the assigned order number

        """
        mask = self.df["OrderNumber"] == int(order_number)
        order = self.df[mask].reset_index(drop=True)
        return order

    def get_first_order(self):
        """
        Returns the first order contained in the order book.

        :return:
            The first order contained in the order book

        """
        if not self.df.empty:
            included_orders = self.df.drop_duplicates(subset=["OrderNumber"])
            order_numbers = included_orders["OrderNumber"].to_list()
            order = self.get_order(order_numbers[0])
            return order

        return self.df

    def get_included_orders(self) -> pd.DataFrame:
        """
        Returns the first items of all orders contained in the order book.

        :return:
            A DataFrame containing the first items of every order contained in the order book

        """
        if not self.df.empty:
            included_orders = self.df.drop_duplicates(subset=["OrderNumber"])
        else:
            included_orders = pd.DataFrame(columns=self.columns)
        return included_orders

    def get_random_order(self) -> pd.DataFrame:
        """
        Returns a random order contained in the order book.

        :return:
            A DataFrame containing a random order contained in the order book

        """
        if not self.df.empty:
            idx = random.randrange(1, len(self.df))
            part_of_order = self.df.iloc[idx]
            entire_order = self.df[self.df.OrderNumber.isin([part_of_order["OrderNumber"]])]
            entire_order = entire_order.reset_index(drop=True)
            return entire_order
        return self.df

    def is_empty(self):
        """
        Returns True if the order book is empty and False if not.

        :return:
            True if the order book is empty and False if not

        """
        if self.df.empty:
            return True
        else:
            return False

    def clear(self) -> None:
        """
        Clears the order intake data frame.

        """
        self.df = self.df[0:0]


class Queue(BaseTemplate):
    """
    This class represents a Queue in a simul8 simulation. It contains information about all order waiting to be
    processed by the machine which the queue is assigned to.

    ...

    Attributes
    ----------
    of_system : str
        Information about the machine that the queue is assigned to
    subsequent: bool
        Information about whether the queue is a subsequent queue
    subsequent_system : str
        Information about the system that follows the system assigned to the queue

    Methods
    -------
    get_state(t_remaining, t_waiting, t_slack,of_n_orders)
        Returns the features that indicate the current state of the orders in the queue
    get_min_max_slack_time()
        Returns  the minimum and maximum slack time from all orders
    sort(heuristic)
        Sorts the queue according to a transferred heuristic
    """

    def __init__(self, problem, of_system: str, subsequent: bool=False, subsequent_system: str=None):
        super().__init__(problem)
        self.of_system = of_system
        self.subsequent = subsequent
        self.subsequent_system = subsequent_system

    def get_state(self, t_remaining: bool=False, t_waiting: bool=False, t_slack: bool=False,
                  of_n_orders: int=None):
        """
        Returns the features that indicate the current state of the orders in the queue.

        Parameters
        ----------
            t_remaining: bool
                True if the remaining processing time is to be calculated from all orders in the queue
            t_waiting: bool
                True if the waiting time is to be calculated from all orders in the queue
            t_slack: bool
                True if the slack time is to be calculated from all orders in the queue
            of_n_orders: int
                Specifies the number of orders for which the transferred values are to be calculated
                (e.g. if of_n_order=5, the values for the first 5 orders in the queue are calculated)

        :return:
            Lists containing the selected values (t_remaining, t_waiting, t_slack) for all (or of_n_orders)
            orders in the queue

        """
        included_orders = self.df.drop_duplicates(subset=["OrderNumber"])
        list_time_remaining = []
        list_waiting_time = []
        list_slack_time = []
        order_numbers = []
        if not included_orders.empty:
            order_numbers = included_orders["OrderNumber"].to_list()
            if of_n_orders and len(order_numbers) >= of_n_orders:
                order_numbers = order_numbers[:of_n_orders]
            for order_number in order_numbers:
                order = self.df[self.df.OrderNumber.isin([order_number])].reset_index(drop=True)
                if t_remaining:
                    tr = time_remaining(order, self.problem.sim.processing_times)
                    list_time_remaining.append(tr)
                if t_waiting:
                    wt = waiting_time(order, self.problem.sim.t_step)
                    list_waiting_time.append(wt)
                if t_slack:
                    st = slack_time(order, self.problem.sim.processing_times, self.problem.sim.t_step)
                    list_slack_time.append(st)

        return list_time_remaining, list_waiting_time, list_slack_time, order_numbers

    def get_min_max_slack_time(self):
        """
        Returns the minimum and maximum slack time from all orders

        :return:
            The minimum and maximum slack time from all orders

        """
        min_st = np.inf
        max_st = -np.inf

        included_orders = self.df.drop_duplicates(subset=["OrderNumber"])
        if not included_orders.empty:
            order_numbers = included_orders["OrderNumber"].to_list()
            for order_number in order_numbers:
                order = self.df[self.df.OrderNumber.isin([order_number])].reset_index(drop=True)
                st = slack_time(order, self.problem.sim.processing_times, self.problem.sim.t_step)
                if st > max_st:
                    max_st = st
                if st < min_st:
                    min_st = st

        return min_st, max_st

    def sort(self, heuristic: str):
        """
        Sorts the queue according to a transferred heuristic

        Parameters
        ----------
            heuristic: str
                The heuristics according to which the queue should be sorted

        """
        included_orders = self.df.drop_duplicates(subset=["OrderNumber"])
        if not included_orders.empty:
            columns = self.columns.copy()
            columns.append("SortBy")
            spt_df = pd.DataFrame(columns=columns)
            order_numbers = included_orders["OrderNumber"].to_list()
            for order_number in order_numbers:
                order = self.df[self.df.OrderNumber.isin([order_number])].reset_index(drop=True)
                sort_by = 0
                if heuristic == "SPT":
                    sort_by = time_remaining(order, self.problem.sim.processing_times)
                elif heuristic == "FIFO":
                    sort_by = order.at[0, "t_slot"]
                elif heuristic == "EDD":
                    sort_by = time_until_due_date(order, self.problem.sim.t_step)
                elif heuristic == "CR":
                    tr = time_remaining(order, self.problem.sim.processing_times)
                    ttd = time_until_due_date(order, self.problem.sim.t_step)
                    sort_by = ttd / tr
                elif heuristic == "SPRPT":
                    tr = time_remaining(order, self.problem.sim.processing_times)
                    st = slack_time(order, self.problem.sim.processing_times, self.problem.sim.t_step)
                    sort_by = st/tr
                order["SortBy"] = sort_by
                spt_df = pd.concat([spt_df, order], ignore_index=True).reset_index(drop=True)
            spt_df = spt_df.sort_values(by=["SortBy", "OrderNumber", "ItemNumber"], ascending=True)
            # with pd.option_context("display.max_rows", None, "display.max_columns", None):
            #     print(spt_df)
            self.df = spt_df.drop(["SortBy"], axis=1)


class SimState(BaseTemplate):
    """
    This class represents the SimState of a simul8 simulation. It contains information about all orders that are
    currently being processed within the production network.

    ...

    Attributes
    ----------
    excel_directory : str
        The directory of the Excel file which is used to share information between Simul8 and Simul8 python API.

    Methods
    -------
    to_excel()
        Writes the information of the SimState to the Excel file stored at the storage location
    from_excel()
        Writes the information of the Excel file stored at the storage location to the SimState DataFrame
    all_waiting()
        Returns True if all orders contained in the SimState DataFrame are waiting to be processed on a subsequent
        machine
    """

    def __init__(self, problem):
        super().__init__(problem)
        file_directory = os.path.dirname(__file__)
        self.excel_directory = os.path.join(file_directory, "State", f"SimState.xlsx")
        self.to_excel()

    def to_excel(self) -> None:
        """
        Writes the information of the SimState to the Excel file stored at the storage location

        """
        self.df.to_excel(self.excel_directory, index=False)

    def from_excel(self) -> None:
        """
        Writes the information of the Excel file stored at the storage location to the SimState DataFrame

        """
        self.df = pd.read_excel(self.excel_directory)

    def all_waiting(self):
        """
        Returns True if all orders contained in the SimState DataFrame are waiting to be processed on a subsequent
        machine

        :return:
            True if all orders contained in the SimState DataFrame are waiting to be processed on a subsequent
            machine and False if not

        """
        if (self.df["InQueue"] == "FD").all():
            return True

        return False


class LabelData:
    """
    This class represents the values that are assigned to the labels of the elements in the Simul8 simulation

    ...

    Attributes
    ----------
    of_system: str
        Information about the machine that the queue is assigned to
    excel_directory  str
        The directory of the Excel file which is used to share information between Simul8 and Simul8 python API.
    columns: list
        A list containing column name
    df : int
        A DataFrame that is used as a memory for the label values

    Methods
    -------
    set_data()
        Transfers the passed values to the DataFrame
    clear()
        Clears the data frame
    to_excel()
        Transfers the label values into the Excel file stored at the storage location
    from_excel()
        Transfers the information in the Excel files to the DataFrame of the class
    """

    def __init__(self, of_system: str):
        self.of_system = of_system
        file_directory = os.path.dirname(__file__)
        self.excel_directory = os.path.join(file_directory, "LabelData", f"LabelData_{of_system}.xlsx")

        self.columns = ["OrderNumber", "S", "M",  "MF", "E", "EH", "VB", "SWD", "FD", "n_Items", "ItemType"]

        self.df = pd.DataFrame(columns=self.columns)
        self.df.to_excel(self.excel_directory, index=False)

    def set_data(self, data: pd.DataFrame) -> None:
        """
        Transfers the passed values to the DataFrame

        Parameters
        ----------
            data: DataFrame
                The values that are transferred to the DataFrame

        """
        self.clear()
        self.df = pd.concat([self.df, data], ignore_index=True)

    def clear(self) -> None:
        """
        Clears the data frame

        """
        self.df = self.df[0:0]

    def to_excel(self) -> None:
        """
        Transfers the label values into the Excel file stored at the storage location

        """
        self.df.to_excel(self.excel_directory, index=False)

    def from_excel(self) -> None:
        """
        Transfers the information in the Excel files to the DataFrame of the class

        """
        self.df = pd.read_excel(self.excel_directory)


class Results(BaseTemplate):
    """
    This class represents a DataFrame in which the training results are written

    ...

    Methods
    -------
    to_excel()
        Transfers the results into the Excel file stored at the storage location
    get()
        Returns the DataFrame containing the training results
    """

    def __init__(self, problem):
        super().__init__(problem)

    def to_excel(self, directory) -> None:
        """
        Transfers the passed values to the DataFrame

        Parameters
        ----------
            directory: str
                The storage path where the result file is to be saved

        """
        self.df.to_excel(directory, index=False)

    def get(self):
        """
        Returns the DataFrame containing the training results

        :return:
            The DataFrame containing the training results

        """
        return self.df


class Simul8Problem:
    """
    This class manages the interactions between the agents and the components of the simulation model stored as
    programme code.

    ...

    Attributes
    ----------
    sim: Simul8API
        An instance of the Simul8API class
    heuristics:  dict
        A dictionary that contains the selectable heuristics in pairs for the indices output by the agents
    queues: list
        A list containing instances of the Queue class that map all queues within the production network
    sim_state: SimState
        An instance of the SimState class
    label_data: list
        A list containing instances of the LabelData class that map all starting queues within the production network
    denominator: None | float
        The value of the denominator in the calculation of the reward
    results: Results
        An instance of the Results class


    Methods
    -------
    _get_queue(of_system)
        Returns a queue instance according to the system to which it is assigned
    -gen_new_work_item(order_number, item_number, item_type, deadline, t_slot, components, on_system)
        Generates a new work item belonging to an order by transferring the information about the work item to a
        DataFrame
    _gen_label_data(order_number, n_items, item_type, components)
        Generates the label values based on the information from the selected order
    append_order_to_queue(of_system, order)
        Generates new work items based on the information from an order, which are assigned to the associated queue
    remove_order_from_queue(queue, order_number)
        Deletes an order based on a transferred order number from the queue in which the order is stored
    remove_order_from_state(order_number)
        Removes an order based on a transferred order number from the SimState DataFrame
    clear_all_queues()
        Clears all queues contained in the queues list
    clear_sim_state()
        Clears the SimState DataFrame
    clear_results()
        Clears the Results DataFrame
    n_orders_in_queue(of_system)
        Returns the number of orders contained in a specific queue
    oi_to_queues(oi)
        Transfers the new incoming orders to the corresponding queues
    set_label_data(of_system, order)
        Transfers the label values related to a selected order to the LabelData DataFrame
    get_random_order_from_queue(of_system)
        Returns a random order from a specific queue
    get_first_order_from_queue(of_system)
        Returns the first order from a specific queue
    get_order_from_queue(of_system, order_number)
        Returns a specific order from a specific queue
    move_to_next_queue(order_number, from_system, to_system)
        Moves an order to the queue of the station at which the next processing steps are to be carried out
    update_state(order)
        Updates the SimState when a job is forwarded to a new queue
    order_to_state(order)
        Transfers the products of a selected order to the SimState
    get_avg_state(of_system)
        Returns the features for the first agent
    get_order_specific_state(of_system, of_n_orders)
        Returns the features for the second agent
    get_avg_slack_time()
        Returns the average slack time calculated from all orders
    get_avg_machine_utilization()
        Returns the average machine utilization calculated from all machines
    save_results(of_system, order_number)
        Saves the results generated after completion of a job in the Results DataFrame
    compute_reward()
        Computes the reward
    sort_queue(of_system, heuristic)
        Sorts a specific queue according to a specific heuristic
    done()
        Returns True if all queues are empty
    order_in_queue(of_system)
        Returns True if a specific queue is not empty
    problem_to_excel()
        Transfers the LabelData and SimState instances to Excel files
    excel_to_problem()
        Transfers the values contained in the associated Excel files to the DataFrames of the LabelData and
        SimState instances
    update_queues()
        Updates the queues based on the SimState
    """

    def __init__(self, sim):
        self.sim = sim
        self.heuristics = {0: "SPT", 1: "EDD", 2: "FIFO", 3: "SPRPT"}

        self.queues = [Queue(self,"16", False, "FD"),
                       Queue(self, "6", False, "FD"),
                       Queue(self, "4", False,"FD"),
                       Queue(self, "25", False, "FD"),
                       Queue(self, "15", False, "FD"),
                       Queue(self,"FD", True)]

        self.sim_state = SimState(self)

        self.label_data = [LabelData("16"),
                           LabelData("6"),
                           LabelData("4"),
                           LabelData("25"),
                           LabelData("15")]

        self.denominator = None

        self.results = Results(self)

    def _get_queue(self, of_system: str) -> Queue:
        """
        Returns a queue instance according to the system to which it is assigned

        Parameters
        ----------
            of_system: str
                The system to which the queue to be returned is assigned to

        :return:
            The queue instance

        """
        for queue in self.queues:
            if queue.of_system == of_system:
                return queue

    @staticmethod
    def _gen_new_work_item(order_number: str, item_number: int, item_type: str, deadline: float, t_slot: float,
                           components: dict, on_system: str) -> pd.DataFrame:
        """
        Generates a new work item belonging to an order by transferring the information about the work item to a
        DataFrame

        Parameters
        ----------
            order_number: str
                The order number of the order to which the new work item belongs
            item_number: int
                The item_number of the new work item
            item_type: str
                The cross-section of the work item
            deadline: float
                The deadline of the order to which the new work item belongs
            t_slot: float
                The point in time at which the order is added to the queue
            components: dict
                A dictionary that indicates the discs that make up the product
            on_system: str
                The queue that currently contains the component

        :return:
            A DataFrame containing the above information

        """
        item = {"OrderNumber": order_number,
                "ItemNumber": item_number,
                "ItemType": item_type,
                "Deadline": deadline,
                "t_slot": t_slot,
                "S": components["S"], "S_Done": 0,
                "M": components["M"], "M_Done": 0,
                "MF": components["MF"], "MF_Done": 0,
                "E": components["E"], "E_Done": 0,
                "EH": components["EH"], "EH_Done": 0,
                "VB": components["VB"], "VB_Done": 0,
                "SWD": components["SWD"], "SWD_Done": 0,
                "VSW": components["VSW"], "VSW_Done": 0,
                "FD": components["FD"], "FD_Done": 0,
                "VFD": components["VFD"], "VFD_Done": 0,
                "InQueue": on_system,
                "t_start": 0,
                "t_start_waiting": 0,
                "t_stop_waiting": 0}
        item = pd.DataFrame([item])
        return item

    @staticmethod
    def _gen_label_data(order_number: str, n_items: str | int, item_type: str | int, components: dict) -> pd.DataFrame:
        """
        Generates the label values based on the information from the selected order

        Parameters
        ----------
            order_number: str
                The order number of the order to which the new work item belongs
            n_items: str
                The number of products included in an order
            item_type: str
                The cross-section of the work item
            components: dict
                A dictionary that indicates the discs that make up the product

        :return:
            A DataFrame containing the above information

        """
        data = {"OrderNumber": order_number,
                "n_Items": n_items,
                "S": components["S"],
                "M": components["M"],
                "MF": components["MF"],
                "E": components["E"],
                "EH": components["EH"],
                "VB": components["VB"],
                "SWD": components["SWD"],
                "FD": components["FD"],
                "ItemType": item_type}
        data = pd.DataFrame([data])
        return data

    def append_order_to_queue(self, of_system: str, order: pd.DataFrame) -> None:
        """
        Generates new work items based on the information from an order, which are assigned to the associated queue

        Parameters
        ----------
            of_system: str
                The system to which the queue to be returned is assigned to
            order: DataFrame
                A DataFrame containing the products of an order

        """
        queue = self._get_queue(of_system)

        order_number = order["OrderNumber"]
        item_type = order["ItemType"]
        t_slot = self.sim.t_step
        deadline = float(self.sim.t_step)+float(order["Deadline"])
        components = {"S": order["S"],
                      "M": order["M"], "MF": order["MF"],
                      "E": order["E"], "EH": order["EH"],
                      "VB": order["VB"],
                      "SWD": order["SWD"], "VSW": 1 if order["SWD"] > 0 else 0,
                      "FD": order["FD"], "VFD": 1 if order["FD"] > 0 else 0}
        on_system = of_system

        for item_number in range(1, order["n_Items"]+1):
            item = self._gen_new_work_item(order_number, item_number, item_type, deadline, t_slot, components, on_system)
            queue.append(item)

    def remove_order_from_queue(self, of_system: str, order_number: str) -> None:
        """
        Deletes an order based on a transferred order number from the queue in which the order is stored

        Parameters
        ----------
            of_system: str
                The system to which the queue to be returned is assigned to
            order_number: str
                The order number of the order to delete

        """
        queue = self._get_queue(of_system)
        queue.remove_order(order_number)

    def remove_order_from_state(self, order_number: str) -> None:
        """
        Removes an order based on a transferred order number from the SimState DataFrame

        Parameters
        ----------
            order_number: str
                The order number of the order to delete

        """
        self.sim_state.remove_order(order_number)

    def clear_all_queues(self) -> None:
        """
        Clears all queues contained in the queues list

        """
        for queue in self.queues:
            queue.clear()

    def clear_sim_state(self) -> None:
        """
        Clears the SimState DataFrame

        """
        self.sim_state.clear()

    def clear_results(self):
        """
        Clears the Results DataFrame

        """
        self.results.clear()

    def n_orders_in_queue(self, of_system):
        """
        Returns the number of orders contained in a specific queue

        Parameters
        ----------
            of_system: str
                The system to which the queue to be returned is assigned to

        :return:
            The number of orders contained in the transferred queue

        """
        queue = self._get_queue(of_system)
        return len(queue)

    def oi_to_queues(self, oi: OrderIntake) -> None:
        """
        Transfers the new incoming orders to the corresponding queues

        Parameters
        ----------
            oi: OrderIntake
                An instance of the OrderIntake class

        """
        for queue in self.queues:
            if not queue.subsequent:
                new_orders = oi.get_new_orders_of_type(queue.of_system)
                if not new_orders.empty:
                    for idx in range(len(new_orders)):
                        new_order = new_orders.iloc[idx]
                        self.append_order_to_queue(queue.of_system, new_order)

    def set_label_data(self, of_system: str, order: pd.DataFrame) -> None:
        """
        Transfers the label values related to a selected order to the LabelData DataFrame

        Parameters
        ----------
            of_system: str
                The system to which the queue to be returned is assigned to
            order: DataFrame
                A DataFrame containing the products of an order

        """
        order_number = order.at[0, "OrderNumber"]
        n_items = len(order)
        item_type = order.at[0, "ItemType"]
        components = {"S": 1 if order.at[0, "S"] > 0 else 0,
                      "M": 1 if order.at[0, "M"] > 0 else 0,
                      "MF": 1 if order.at[0, "MF"] > 0 else 0,
                      "E": 1 if order.at[0, "E"] > 0 else 0,
                      "EH": 1 if order.at[0, "EH"] > 0 else 0,
                      "VB": 1 if order.at[0, "VB"] > 0 else 0,
                      "SWD": 1 if order.at[0, "SWD"] > 0 else 0,
                      "FD": 1 if order.at[0, "FD"] > 0 else 0}
        data = self._gen_label_data(order_number, n_items, item_type, components)
        for label_data in self.label_data:
            if label_data.of_system == of_system:
                label_data.set_data(data)

    def get_random_order_from_queue(self, of_system):
        """
        Returns a random order from a specific queue

        Parameters
        ----------
            of_system: str
                The system to which the queue to be returned is assigned to

        :return:
            A random order

        """
        queue = self._get_queue(of_system)
        order = queue.get_random_order()
        return order

    def get_first_order_from_queue(self, of_system):
        """
        Returns the first order from a specific queue

        Parameters
        ----------
            of_system: str
                The system to which the queue to be returned is assigned to

        :return:
            The first order

        """
        queue = self._get_queue(of_system)
        order = queue.get_first_order()
        return order

    def get_order_from_queue(self, of_system, order_number):
        """
        Returns a specific order from a specific queue

        Parameters
        ----------
            of_system: str
                The system to which the queue to be returned is assigned to
            order_number: str
                The order number of the order to return


        :return:
            The order associated to the transferred order number

        """
        def compute_denominator_for_reward():
            min_st, max_st = queue.get_min_max_slack_time()
            order_st = slack_time(order, self.sim.processing_times, self.sim.t_step)
            diff = max_st - min_st
            if not diff == 0:
                norm_st = (1 / diff) * (order_st - min_st)
            else:
                norm_st = 1
            self.denominator = round(1 / (1 + np.exp(-10 * (norm_st - 0.5))), 5)

        queue = self._get_queue(of_system)
        order = queue.get_order(order_number)
        compute_denominator_for_reward()
        return order

    def move_to_next_queue(self, order_number, from_system, to_system):
        """
        Moves an order to the queue of the station at which the next processing steps are to be carried out

        Parameters
        ----------
            order_number: str
                The order number of the order to move
            from_system: str
                The system to which the order has to be moved
            to_system: str
                The system from which the order is moved

        """
        source_queue = self._get_queue(from_system)
        order = source_queue.get_order(order_number)
        target_queue = self._get_queue(to_system)
        order["t_start_waiting"] = self.sim.t_step
        order["InQueue"] = to_system
        target_queue.append(order)
        source_queue.remove_order(order_number)
        self.sim_state.remove_order(order_number)
        self.sim_state.append(order)

    def update_state(self, order):
        """
        Updates the SimState when a job is forwarded to a new queue

        Parameters
        ----------
             order: DataFrame
                A DataFrame containing the products of an order

        """
        order_number = order.at[0, "OrderNumber"]
        self.sim_state.get_order(order_number)
        order["t_stop_waiting"] = self.sim.t_step
        self.sim_state.remove_order(order_number)
        self.sim_state.append(order)

    def order_to_state(self, order: pd.DataFrame) -> None:
        """
        Transfers the products of a selected order to the SimState

        Parameters
        ----------
             order: DataFrame
                A DataFrame containing the products of an order

        """
        order["t_start"] = self.sim.t_step
        self.sim_state.append(order)

    def get_avg_state(self, of_system):
        """
        Returns the features for the first agent

        Parameters
        ----------
             of_system: str
                The system to which the queue is assigned, from whose products the features are calculated

        :return:
            The features for the first agent

        """
        def calc_avg_state_values(lst_tr, lst_tw, lst_st):
            avg_time_remaining, avg_time_waiting, avg_slack_time = 0, 0, 0
            if lst_tr:
                avg_time_remaining = round(sum(lst_tr) / len(lst_tr), 3)
            if lst_tw:
                avg_time_waiting = round(sum(lst_tw) / len(lst_tw), 3)
            if lst_st:
                avg_slack_time = round(sum(lst_st) / len(lst_st), 3)

            return avg_time_remaining, avg_time_waiting, avg_slack_time

        queue = self._get_queue(of_system)
        lst_time_remaining, lst_time_waiting, lst_slack_time, _ = queue.get_state(t_remaining=True, t_waiting=True,
                                                                                  t_slack=True)

        lst_time_remaining_sub, lst_time_waiting_sub, lst_slack_time_sub = [], [], []
        if queue.subsequent_system:
            subsequent_queue = self._get_queue(queue.subsequent_system)
            lst_time_remaining_sub, lst_time_waiting_sub, lst_slack_time_sub, _ = subsequent_queue.get_state(t_remaining=True,
                                                                                                          t_waiting=True,
                                                                                                          t_slack=True)


        avg_tr, avg_tw, avg_st = calc_avg_state_values(lst_time_remaining, lst_time_waiting, lst_slack_time)

        avg_tr_sub, avg_wt_sub, avg_st_sub = calc_avg_state_values(lst_time_remaining_sub, lst_time_waiting_sub,
                                                                   lst_slack_time_sub)

        np_current_state = np.array([avg_tr, avg_tw, avg_st, avg_tr_sub, avg_wt_sub, avg_st_sub])

        return np_current_state

    def get_order_specific_state(self, of_system, of_n_orders: int):
        """
        Returns the features for the second agent

        :return:
            The features for the second agent

        """
        queue = self._get_queue(of_system)
        lst_time_remaining, lst_time_waiting, lst_slack_time, order_numbers = queue.get_state(t_remaining=True,
                                                                                              t_waiting=True,
                                                                                              t_slack=True,
                                                                                              of_n_orders=of_n_orders)

        lst_time_remaining_sub= []
        if queue.subsequent_system:
            subsequent_queue = self._get_queue(queue.subsequent_system)
            lst_time_remaining_sub, _, _, _ = subsequent_queue.get_state(t_remaining=True,
                                                                         of_n_orders=of_n_orders)

        data = []
        for i in range(of_n_orders):
            try:
                data.append(lst_time_remaining[i])
                data.append(lst_time_waiting[i])
                data.append(lst_slack_time[i])
                if queue.subsequent_system:
                    order = queue.get_order(order_numbers[i])
                    if (order[queue.subsequent_system] > 0).all():
                        data.append(1)
                    else:
                        data.append(0)
                    if len(lst_time_remaining_sub) > 0:
                        avg_tr_sub = round(sum(lst_time_remaining_sub) / len(lst_time_remaining_sub), 3)
                        data.append(avg_tr_sub)
                    else:
                        data.append(0)
                else:
                    data.extend([0, 0])
            except IndexError:
                data.extend([0, 0, 0, 0, 0])

        np_current_state = np.array(data)

        return np_current_state, order_numbers

    def get_avg_slack_time(self):
        """
        Returns the average slack time calculated from all orders

        :return:
            The average slack time

        """

        complete_list_slack_time = []
        for queue in self.queues:
            _, _, lst_slack_time, _ = queue.get_state(t_slack=True)
            if lst_slack_time:
                complete_list_slack_time.extend(lst_slack_time)

        if len(complete_list_slack_time) > 1:
            avg_st = sum(complete_list_slack_time) / len(complete_list_slack_time)
            avg_st = round(avg_st, 3)
        else:
            avg_st = complete_list_slack_time[0]

        return avg_st

    def get_avg_machine_utilization(self):
        """
        Returns the average machine utilization

        :return:
            The average machine utilization

        """
        orders_on_shop_floor = self.sim_state.get_included_orders()
        order_numbers = orders_on_shop_floor["OrderNumber"].to_list()
        system_related_utilizations = []
        for order_number in order_numbers:
            order = self.sim_state.get_order(order_number)
            system_related_utilization = machine_utilization(order, self.sim.t_step)
            system_related_utilizations.append(system_related_utilization)
        avg_system_utilization = round(sum(system_related_utilizations) / len(system_related_utilizations), 3)

        return avg_system_utilization

    def save_results(self, of_system, order_number):
        """
        Saves the results generated after completion of a job in the Results DataFrame

        Parameters
        ----------
             of_system: str
                The system to which the queue is assigned, whose products results have to be saved
            order_number: str
                The order number of the order for which the results have to be saved

        """
        queue = self._get_queue(of_system)
        order = queue.get_order(order_number)
        order["t_stop"] = self.sim.t_step
        self.results.append(order)

    def compute_reward(self):
        """
        Computes the reward

        :return:
            The value of the reward

        """
        avg_machine_utilization = self.get_avg_machine_utilization()
        numerator_reward = round(1 / (1 + np.exp(-10 * (avg_machine_utilization - 0.5))), 5)

        denominator_reward = self.denominator
        if math.isnan(denominator_reward):
            denominator_reward = 1
            print("??????????????????????????????????????????? NAN VALUE ???????????????????????????????????????????")

        print(f"Numerator: {numerator_reward}; Denominator: {denominator_reward}")

        reward = round(numerator_reward / denominator_reward, 3)

        return reward

    def sort_queue(self, of_system, heuristic):
        """
        Sorts a specific queue according to a specific heuristic

        """
        queue = self._get_queue(of_system)
        queue.sort(heuristic)

    def done(self):
        """
        Returns True if all queues are empty

        :return:
            True if all queues are empty and False if not

        """
        for queue in self.queues:
            if not queue.is_empty():
                return False

        return True

    def order_in_queue(self, of_system):
        """
        Returns True if a specific queue is not empty

        Parameters
        ----------
             of_system: str
                The system to which the queue is assigned for which you want to check whether it is empty

        :return:
            True if the specific queue is empty and False if not

        """
        queue = self._get_queue(of_system)
        if not queue.is_empty():
            return True

        return False

    def problem_to_excel(self) -> None:
        """
        Transfers the LabelData and SimState instances to Excel files

        """
        self.sim_state.to_excel()

        for label_data in self.label_data:
            label_data.to_excel()

    def excel_to_problem(self) -> None:
        """
        Transfers the values contained in the associated Excel files to the DataFrames of the LabelData and
        SimState instances

        """
        self.sim_state.from_excel()

        for label_data in self.label_data:
            label_data.from_excel()

    def update_queues(self):
        """
        Updates the queues based on the SimState

        """
        orders_on_system = self.sim_state.get_included_orders()
        order_numbers_of_orders_on_system = orders_on_system["OrderNumber"].tolist()
        for order_number in order_numbers_of_orders_on_system:
            order = self.sim_state.get_order(order_number)
            queue = self._get_queue(of_system=f'{order.at[0, "InQueue"]}')
            queue.remove_order(order_number)
            queue.append(order)


class Environment:
    """
    This class simulates the environment from which the Simul8API and the Simul8 problem are accessed

    ...

    Attributes
    ----------
    sim: Simul8API
        An instance of the Simul8API class
    problem:  Simul8Problem
        An instance of the Simul8Problem class
    """

    def __init__(self):
        self.sim = Simul8API(environment=self)
        self.problem = Simul8Problem(sim=self.sim)
