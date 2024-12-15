from OrderIntake.OrderManager import OrderIntake, RandomOrder
from Simulation.S8Environment import Environment
from Agent.PPOAgent import Agent
import os
import random
import pythoncom
import traceback
import pandas as pd
from typing_extensions import Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    """
    This function was taken from the Matplotlib Tutorial - Radar chart
    @website{
    author={The Matplotlib development team}
    title={Radar chart (aka spider or star chart)}
    year={w. d.}
    url={https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html}
    }

    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def adjust_length(history_dict: dict):
    max_len = 0
    for arr in history_dict.values():
        if len(arr) > max_len:
            max_len = len(arr)

    for key in history_dict.keys():
        diff = max_len - len(history_dict[key])
        if diff > 0:
            for i in range(diff):
                history_dict[key].append(0)

    return history_dict

def test_loop_agent(environment):
    """
    A loop to test the fully trained DRL agents.

    Parameters
    ----------
    environment: Environment
        An instance of the Environment class

    """

    print("--> Instantiate Agents")
    actor_agent_1 = os.path.join(os.getcwd(), "Agent", "Model_Stage_1", "Run 10", "ActorEP28.pth")
    number_of_selectable_heuristics = 4
    agent_stage_1 = Agent(n_actions=number_of_selectable_heuristics, n_states=6, alpha=0.0002, beta=0.0005, gamma=0.99,
                    lambd=0.95, clip=0.2, batch_size=512, actor_model=actor_agent_1)
    actor_agent_2 = os.path.join(os.getcwd(), "Agent", "Model_Stage_2", "Run 10", "ActorEP28.pth")
    number_of_selectable_orders = 5
    agent_stage_2 = Agent(n_actions=number_of_selectable_orders, n_states=number_of_selectable_orders * 5, alpha=0.0002,
                    beta=0.0005, gamma=0.99, lambd=0.95, clip=0.2, batch_size=512, actor_model=actor_agent_2)

    avg_slack_time_history = {"Avg Slack Time": []}
    avg_utilization_history = {"Avg Machine Utilization": []}

    done = False
    new_decision = False

    print("--> Set the starting Orders to the simulation queues")
    for queue in environment.problem.queues:
        of_sys = queue.of_system
        if environment.problem.order_in_queue(of_sys):

            current_state_1 = environment.problem.get_avg_state(of_sys)
            action_1, log_prob_1 = agent_stage_1.get_action(current_state_1)
            heuristic = environment.problem.heuristics[action_1]
            print(f"    Heuristic: {heuristic}")
            environment.problem.sort_queue(of_sys, heuristic)

            current_state_2, lst_order_numbers = environment.problem.get_order_specific_state(of_sys,
                                                                                              of_n_orders=number_of_selectable_orders)
            action_2, log_prob_2 = agent_stage_2.get_action(current_state_2, masked=True)
            nb_chosen_order = lst_order_numbers[action_2]
            print(f"    Chosen Order: {nb_chosen_order}")

            initial_order = environment.problem.get_order_from_queue(of_sys, nb_chosen_order)

            # initial_order = environment.problem.get_random_order_from_queue(of_sys)
            environment.problem.order_to_state(initial_order)
            environment.problem.set_label_data(of_system=of_sys, order=initial_order)
            environment.sim.wi_to_queue(nb_wi=len(initial_order), queue=of_sys)

    environment.problem.problem_to_excel()
    environment.sim.excel_to_sim()
    print("--> Start Simulation")
    environment.sim.run(0, sim_speed=100)

    environment.sim.listenForMessages = True
    try:
        # Start loop to listen for events
        while environment.sim.listenForMessages and not done:
            # Process messages and return as soon as there are no more waiting messages to process
            pythoncom.PumpWaitingMessages()

            if environment.sim.new_request:
                print("\n--> New Request")
                environment.sim.sim_to_excel()
                environment.problem.excel_to_problem()
                environment.problem.update_queues()

                requests = environment.sim.get_request()

                for request in requests:
                    command = request["Command"]
                    attributes = request["Attributes"]
                    # Command: MoveOn, Attributes: FromSystem, ToSystem, OrderNumber
                    if command == "MoveOn":
                        from_system = attributes[0]
                        to_system = attributes[1]
                        order_number = attributes[2]
                        print(f"\n--> Request: \n    Command: {command}, FromSystem: {from_system}, ToSystem: {to_system}, OrderNumber: {order_number}")
                        environment.problem.move_to_next_queue(f"{order_number}", f"{from_system}", f"{to_system}")
                    # Command: DeleteOrder, Attributes: FromQueue, OrderNumber
                    elif command == "DeleteOrder":
                        from_system = attributes[0]
                        order_number = attributes[1]
                        print(
                            f"\n--> Request: \n    Command: {command}, FromSystem: {from_system}, OrderNumber: {order_number}")
                        environment.problem.save_results(from_system, order_number)
                        environment.problem.remove_order_from_queue(f"{from_system}", f"{order_number}")
                        environment.problem.remove_order_from_state(f"{order_number}")
                    # Command: NewOrder, Attributes: ToSystem
                    elif command == "NewOrder":
                        to_system = attributes[0]
                        print(f"\n--> Request: \n    Command: {command}, ToSystem: {to_system}")

                        if new_decision:
                            avg_machine_utilization = environment.problem.get_avg_machine_utilization()
                            print(f"    Avg. Machine Utilization: {avg_machine_utilization}")
                            avg_utilization_history["Avg Machine Utilization"].append(avg_machine_utilization)

                            avg_slack_time = environment.problem.get_avg_slack_time()
                            print(f"    Avg. Slack Time: {avg_slack_time}")
                            avg_slack_time_history["Avg Slack Time"].append(avg_slack_time)

                        if environment.problem.order_in_queue(to_system):
                            new_decision = True
                            current_state_1 = environment.problem.get_avg_state(to_system)
                            action_1, log_prob_1 = agent_stage_1.get_action(current_state_1)
                            heuristic = environment.problem.heuristics[action_1]
                            print(f"    Heuristic: {heuristic}")
                            environment.problem.sort_queue(to_system, heuristic)

                            current_state_2, lst_order_numbers = environment.problem.get_order_specific_state(to_system,
                                                                                                      of_n_orders=number_of_selectable_orders)
                            action_2, log_prob_2 = agent_stage_2.get_action(current_state_2, masked=True)
                            nb_chosen_order = lst_order_numbers[action_2]
                            print(f"    Chosen Order: {nb_chosen_order}")

                            chosen_order = environment.problem.get_order_from_queue(to_system, nb_chosen_order)
                            if to_system == "FD":
                                environment.problem.update_state(chosen_order)
                                sys = chosen_order.at[0, "ItemType"]
                                environment.sim.link_sim_objects(f"{sys}")
                            else:
                                environment.problem.order_to_state(chosen_order)
                                environment.problem.set_label_data(of_system=to_system, order=chosen_order)
                                environment.sim.wi_to_queue(nb_wi=len(chosen_order), queue=to_system)
                        else:
                            new_decision = False

                if environment.problem.done():
                    done = True
                    environment.sim.new_request = False
                else:
                    environment.problem.problem_to_excel()
                    environment.sim.excel_to_sim()
                    environment.sim.new_request = False
                    environment.sim.run(0, sim_speed=100)

    except Exception:
        print(traceback.format_exc())

    order_related_results_directory = os.path.join(memory_path_results, "OrderRelatedResults_Agent.xlsx")
    environment.problem.results.to_excel(directory=order_related_results_directory)

    avg_slack_time_history = adjust_length(avg_slack_time_history)
    avg_utilization_history = adjust_length(avg_utilization_history)

    df = pd.DataFrame(avg_slack_time_history)
    st_directory = os.path.join(memory_path_results, f"AvgSlackTime_Agent.xlsx")
    df.to_excel(st_directory, index=False)

    df = pd.DataFrame(avg_utilization_history)
    ut_directory = os.path.join(memory_path_results, f"AvgMachineUtilization_Agent.xlsx")
    df.to_excel(ut_directory, index=False)

    return environment.problem.results.get()

def test_loop_heuristic(environment, heuristic):
    """
    A loop for testing certain heuristics based on a Simul8 simulation model.

    Parameters
    ----------
    environment: Environment
        An instance of the Environment class
    heuristic: str
        The heuristic according to which the queues of the requesting systems are sorted for testing

    """
    avg_slack_time_history = {"Avg Slack Time": []}
    avg_utilization_history = {"Avg Machine Utilization": []}

    done = False
    step = 0
    new_decision = False

    print("--> Set the starting Orders to the simulation queues")
    for queue in environment.problem.queues:
        of_sys = queue.of_system
        if environment.problem.order_in_queue(of_sys):
            environment.problem.sort_queue(of_sys, heuristic)
            order = environment.problem.get_first_order_from_queue(of_sys)
            environment.problem.order_to_state(order)
            environment.problem.set_label_data(of_system=of_sys, order=order)
            environment.sim.wi_to_queue(nb_wi=len(order), queue=of_sys)

    environment.problem.problem_to_excel()
    environment.sim.excel_to_sim()
    print("--> Start Simulation")
    environment.sim.run(0, sim_speed=100)

    environment.sim.listenForMessages = True
    try:
        # Start loop to listen for events
        while environment.sim.listenForMessages and not done:
            # Process messages and return as soon as there are no more waiting messages to process
            pythoncom.PumpWaitingMessages()

            if environment.sim.new_request:
                step += 1
                print("\n--> New Request")
                environment.sim.sim_to_excel()
                environment.problem.excel_to_problem()
                environment.problem.update_queues()

                requests = environment.sim.get_request()

                for request in requests:
                    command = request["Command"]
                    attributes = request["Attributes"]
                    # Command: MoveOn, Attributes: FromSystem, ToSystem, OrderNumber
                    if command == "MoveOn":
                        from_system = attributes[0]
                        to_system = attributes[1]
                        order_number = attributes[2]
                        print(
                            f"\n--> Request: \n    Command: {command}, FromSystem: {from_system}, ToSystem: {to_system}, OrderNumber: {order_number}")
                        environment.problem.move_to_next_queue(f"{order_number}", f"{from_system}", f"{to_system}")
                    # Command: DeleteOrder, Attributes: FromQueue, OrderNumber
                    elif command == "DeleteOrder":
                        from_system = attributes[0]
                        order_number = attributes[1]
                        print(
                            f"\n--> Request: \n    Command: {command}, FromSystem: {from_system}, OrderNumber: {order_number}")
                        environment.problem.save_results(from_system, order_number)
                        environment.problem.remove_order_from_queue(f"{from_system}", f"{order_number}")
                        environment.problem.remove_order_from_state(f"{order_number}")
                    # Command: NewOrder, Attributes: ToSystem
                    elif command == "NewOrder":
                        to_system = attributes[0]
                        print(f"\n--> Request: \n    Command: {command}, ToSystem: {to_system}")

                        if new_decision:
                            avg_machine_utilization = environment.problem.get_avg_machine_utilization()
                            print(f"    Avg. Machine Utilization: {avg_machine_utilization}")
                            avg_utilization_history["Avg Machine Utilization"].append(avg_machine_utilization)

                            avg_slack_time = environment.problem.get_avg_slack_time()
                            print(f"    Avg. Slack Time: {avg_slack_time}")
                            avg_slack_time_history["Avg Slack Time"].append(avg_slack_time)

                        if environment.problem.order_in_queue(to_system):
                            new_decision = True
                            environment.problem.sort_queue(to_system, heuristic)
                            chosen_order = environment.problem.get_first_order_from_queue(to_system)
                            if to_system == "FD":
                                environment.problem.update_state(chosen_order)
                                sys = chosen_order.at[0, "ItemType"]
                                environment.sim.link_sim_objects(f"{sys}")
                            else:
                                environment.problem.order_to_state(chosen_order)
                                environment.problem.set_label_data(of_system=to_system, order=chosen_order)
                                environment.sim.wi_to_queue(nb_wi=len(chosen_order), queue=to_system)
                        else:
                            new_decision = False

                if environment.problem.done():
                    done = True
                    environment.sim.new_request = False
                else:
                    environment.problem.problem_to_excel()
                    environment.sim.excel_to_sim()
                    environment.sim.new_request = False
                    environment.sim.run(0, sim_speed=100)

    except Exception:
        print(traceback.format_exc())

    order_related_results_directory = os.path.join(memory_path_results, f"OrderRelatedResults_{heuristic}.xlsx")
    environment.problem.results.to_excel(directory=order_related_results_directory)

    avg_slack_time_history = adjust_length(avg_slack_time_history)
    avg_utilization_history = adjust_length(avg_utilization_history)

    df = pd.DataFrame(avg_slack_time_history)
    st_directory = os.path.join(memory_path_results, f"AvgSlackTime_{heuristic}.xlsx")
    df.to_excel(st_directory, index=False)

    df = pd.DataFrame(avg_utilization_history)
    ut_directory = os.path.join(memory_path_results, f"AvgMachineUtilization_{heuristic}.xlsx")
    df.to_excel(ut_directory, index=False)

    return environment.problem.results.get()


class Evaluation:
    """
    A class for analysing the test results based on various characteristics.

    """

    def __init__(self, data_frame: pd.DataFrame, excel_file: str=None):
        self.df = data_frame
        if excel_file:
            self.df = pd.read_excel(excel_file)

        self.results = {"Avg. Wait Time Queue": [],
                        "Wait Time Queue": [],
                        "Avg. Delay Time": [],
                        "Avg. Nb Delayed Jobs": [],
                        "Nb. Delayed Jobs": [],
                        "Time Delay": [],
                        "Makespan": [],
                        "Avg. Machine Utilization": []}

    @staticmethod
    def _hours(windows_time):
        hours = windows_time * 24
        return hours

    @staticmethod
    def _minutes(windows_time):
        minutes = windows_time * 24 * 60
        return minutes

    @staticmethod
    def _seconds(windows_time):
        seconds = windows_time * 24 * 60 * 60
        return seconds

    def time_convert(self, time_unit, argument):
        match time_unit:
            case "std":
                argument = self._hours(argument)
            case "min":
                argument = self._minutes(argument)
            case "sec":
                argument = self._seconds(argument)

        return argument

    def _iter_through_orders(self):
        included_orders = self.df.drop_duplicates(subset=["OrderNumber"])
        included_order_numbers = included_orders["OrderNumber"].to_list()
        for order_number in included_order_numbers:
            mask = self.df["OrderNumber"] == int(order_number)
            order = self.df[mask].reset_index(drop=True)
            yield order

    def _queue_waiting_time(self):
        results = []
        for idx, order in enumerate(self._iter_through_orders()):
            t_slot = order.at[0, "t_slot"]
            t_start = order.at[0, "t_start"]
            t_wait_in_queue = t_start - t_slot
            results.append(t_wait_in_queue)

        return results

    def calc_avg_queue_waiting_time(self, time_unit: Literal["std", "min", "sec"]):
        order_related_queue_waiting_time = self._queue_waiting_time()
        avg_queue_time_waiting = round(sum(order_related_queue_waiting_time) / len(order_related_queue_waiting_time), 3)
        avg_queue_time_waiting = self.time_convert(time_unit, avg_queue_time_waiting)

        return avg_queue_time_waiting

    def calc_sum_queue_waiting_time(self, time_unit: Literal["std", "min", "sec"]):
        order_related_queue_waiting_time = self._queue_waiting_time()
        sum_queue_time_waiting = sum(order_related_queue_waiting_time)
        sum_queue_time_waiting = self.time_convert(time_unit, sum_queue_time_waiting)

        return sum_queue_time_waiting

    def _time_delay(self):
        results = []
        for idx, order in enumerate(self._iter_through_orders()):
            t_stop = order.at[0, "t_stop"]
            deadline = order.at[0, "Deadline"]
            delay_time = t_stop - deadline
            if delay_time < 0:
                delay_time = 0
            results.append(delay_time)

        return results

    def calc_avg_time_delay(self, time_unit: Literal["std", "min", "sec"]):
        order_related_time_delay = self._time_delay()
        avg_time_delay = round(sum(order_related_time_delay) / len(order_related_time_delay), 3)
        avg_time_delay = self.time_convert(time_unit, avg_time_delay)

        return avg_time_delay

    def calc_avg_number_of_jobs_delayed(self):
        order_related_time_delay = self._time_delay()
        avg_nb_of_jobs_delayed = round(sum(i != 0 for i in order_related_time_delay) / len(order_related_time_delay))

        return avg_nb_of_jobs_delayed

    def calc_number_of_delayed_jobs(self):
        order_related_time_delay = self._time_delay()
        nb_of_delayed_jobs = sum(i != 0 for i in order_related_time_delay)

        return nb_of_delayed_jobs

    def calc_sum_time_delay(self, time_unit: Literal["std", "min", "sec"]):
        order_related_time_delay = self._time_delay()
        time_delay = sum(order_related_time_delay)
        time_delay = self.time_convert(time_unit, time_delay)

        return time_delay

    def calc_make_span(self, time_unit: Literal["std", "min", "sec"]):
        t_stop_max = self.df["t_stop"].max()
        t_start_min = self.df["t_start"].min()
        make_span = round(t_stop_max - t_start_min, 3)
        make_span = self.time_convert(time_unit, make_span)

        return make_span

    def calc_avg_machine_utilization(self):
        results = []
        for idx, order in enumerate(self._iter_through_orders()):
            t_start = order.at[0, "t_start"]
            t_stop = order.at[0, "t_stop"]
            t_start_waiting = order.at[0, "t_start_waiting"]
            t_stop_waiting = order.at[0, "t_stop_waiting"]

            if not t_start_waiting == 0 and not t_stop_waiting == 0:
                utilization = round(1 - ((t_stop - t_start_waiting) - (t_stop - t_stop_waiting)) / (t_stop - t_start), 3)
            else:
                utilization = 1.0
            results.append(utilization)

        avg_machine_utilization = round(sum(results) / len(results), 3)

        return avg_machine_utilization

    def evaluate(self, container: dict=None):
        avg_time_waiting_in_queue = self.calc_avg_queue_waiting_time(time_unit="std")
        print(f"Avg. Queue Waiting Time : {avg_time_waiting_in_queue}")
        sum_time_waiting_in_queue = self.calc_sum_queue_waiting_time(time_unit="std")
        print(f"Queue Waiting Time : {sum_time_waiting_in_queue}")
        avg_delay_time = self.calc_avg_time_delay(time_unit="std")
        print(f"Avg. Delay Time: {avg_delay_time}")
        avg_nb_of_delayed_jobs = self.calc_avg_number_of_jobs_delayed()
        print(f"Avg. Number of Delayed Jobs: {avg_nb_of_delayed_jobs}")
        nb_of_delayed_jobs = self.calc_number_of_delayed_jobs()
        print(f"Number of Delayed Jobs: {nb_of_delayed_jobs}")
        time_delay = self.calc_sum_time_delay(time_unit="std")
        print(f"Time Delay: {time_delay}")
        make_span = self.calc_make_span(time_unit="std")
        print(f"Make Span: {make_span}")
        avg_machine_utilization = self.calc_avg_machine_utilization()
        print(f"Avg. Machine Utilization: {avg_machine_utilization}")

        self.results["Avg. Wait Time Queue"].append(avg_time_waiting_in_queue)
        self.results["Wait Time Queue"].append(sum_time_waiting_in_queue)
        self.results["Avg. Delay Time"].append(avg_delay_time)
        self.results["Avg. Nb Delayed Jobs"].append(avg_nb_of_delayed_jobs)
        self.results["Nb. Delayed Jobs"].append(nb_of_delayed_jobs)
        self.results["Time Delay"].append(time_delay)
        self.results["Makespan"].append(make_span)
        self.results["Avg. Machine Utilization"].append(avg_machine_utilization)

        if container:
            container["Avg. Wait Time Queue"].append(avg_time_waiting_in_queue)
            container["Wait Time Queue"].append(sum_time_waiting_in_queue)
            container["Avg. Delay Time"].append(avg_delay_time)
            container["Avg. Nb Delayed Jobs"].append(avg_nb_of_delayed_jobs)
            container["Nb. Delayed Jobs"].append(nb_of_delayed_jobs)
            container["Time Delay"].append(time_delay)
            container["Makespan"].append(make_span)
            container["Avg. Machine Utilization"].append(avg_machine_utilization)

            return container
        else:
            return self.results


if __name__=="__main__":
    memory_path_results = os.path.join(os.path.dirname(__file__), "Results", "Test")
    results_dict = {"Avg. Wait Time Queue": [],
                    "Wait Time Queue": [],
                    "Avg. Delay Time": [],
                    "Avg. Nb Delayed Jobs": [],
                    "Nb. Delayed Jobs": [],
                    "Time Delay": [],
                    "Makespan": [],
                    "Avg. Machine Utilization": []}

    print("--> Instantiate Environment")
    env = Environment()
    path = os.path.join(os.getcwd(), "Simulation", "JobShop.S8")
    print("--> Open Sim")
    env.sim.open(path)
    print("--> Instantiate Order Intake")
    oi = OrderIntake()
    print("--> Clear Order Intake")
    oi.clear()

    n_default_orders = 120
    print("\n--> Generate Start Orders")
    for _ in range(n_default_orders):
        new_order = RandomOrder()
        data = new_order.get_data()
        if not oi.duplicat(data["OrderNumber"]):
            oi.append(data)

    print(f"\n\n------------------------------ START RUN USING AGENT ------------------------------\n")
    print("--> Clear Queues")
    env.problem.clear_all_queues()
    print("--> Clear Sim State")
    env.problem.clear_sim_state()
    print("--> Clear Results")
    env.problem.clear_results()
    print("--> Problem to Excel")
    env.problem.problem_to_excel()
    print("--> Insert orders to Queues")
    env.problem.oi_to_queues(oi)
    print("--> Start Test")
    results_df = test_loop_agent(env, memory_path_results)
    print("--> Reset")
    env.sim.reset()
    oi.reset_incoming_orders()
    evaluation = Evaluation(results_df)
    results_dict = evaluation.evaluate(results_dict)

    print(f"\n\n------------------------------ START RUN USING FIFO ------------------------------\n")
    print("--> Clear Queues")
    env.problem.clear_all_queues()
    print("--> Clear Sim State")
    env.problem.clear_sim_state()
    print("--> Clear Results")
    env.problem.clear_results()
    print("--> Problem to Excel")
    env.problem.problem_to_excel()
    print("--> Insert orders to Queues")
    env.problem.oi_to_queues(oi)
    print("--> Start Test")
    results_df = test_loop_heuristic(env, "FIFO", memory_path_results)
    print("--> Reset")
    env.sim.reset()
    oi.reset_incoming_orders()
    evaluation = Evaluation(results_df)
    results_dict = evaluation.evaluate(results_dict)

    print(f"\n\n------------------------------ START RUN USING SPT ------------------------------\n")
    print("--> Clear Queues")
    env.problem.clear_all_queues()
    print("--> Clear Sim State")
    env.problem.clear_sim_state()
    print("--> Clear Results")
    env.problem.clear_results()
    print("--> Problem to Excel")
    env.problem.problem_to_excel()
    print("--> Insert orders to Queues")
    env.problem.oi_to_queues(oi)
    print("--> Start Test")
    results_df = test_loop_heuristic(env, "SPT", memory_path_results)
    print("--> Reset")
    env.sim.reset()
    oi.reset_incoming_orders()
    evaluation = Evaluation(results_df)
    results_dict = evaluation.evaluate(results_dict)

    print(f"\n\n------------------------------ START RUN USING SPRPT ------------------------------\n")
    print("--> Clear Queues")
    env.problem.clear_all_queues()
    print("--> Clear Sim State")
    env.problem.clear_sim_state()
    print("--> Clear Results")
    env.problem.clear_results()
    print("--> Problem to Excel")
    env.problem.problem_to_excel()
    print("--> Insert orders to Queues")
    env.problem.oi_to_queues(oi)
    print("--> Start Test")
    results_df = test_loop_heuristic(env, "SPRPT", memory_path_results)
    print("--> Reset")
    env.sim.reset()
    oi.reset_incoming_orders()
    evaluation = Evaluation(results_df)
    results_dict = evaluation.evaluate(results_dict)

    print("\n\n--> Close Simulation")
    env.sim.close()

    varlabels = ["Wait Time Queue", "Avg. Delay Time", "Nb. Delayed Jobs", "Makespan", "Avg. Machine Utilization"]

    evaluation_data = {"DRL-Agent": [],
                       "FIFO": [],
                       "SPT": [],
                       "SPRPT": []}
    types = list(evaluation_data.keys())
    for label, lst in results_dict.items():
        print(f"Irrelative: {label} {lst}")
        if label in varlabels:
            min_value = min(lst)
            max_value = max(lst)
            diff = max_value - min_value
            for idx, value in enumerate(lst):
                if not diff == 0:
                    if not label == "Avg. Machine Utilization":
                        value_ = 1 - (1 / diff) * (value - min_value)
                    else:
                        value_ = (1 / diff) * (value - min_value)
                else:
                    value_ = 1
                lst[idx] = value_
                evaluation_data[types[idx]].append(value_)
            print(f"Relative: {label} {lst}")

    N = len(varlabels)
    theta = radar_factory(N, frame='polygon')

    spoke_labels = varlabels
    title, case_data = "Evaluation", list(evaluation_data.values())

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title, position=(0.5, 1.1), ha='center')

    colors = ['b', 'r', 'g', 'y']
    for d, c in zip(case_data, colors):
        line = ax.plot(theta, d, color=c)
        ax.fill(theta, d, facecolor=c, alpha=0.25, label='_nolegend_')
    ax.set_varlabels(spoke_labels)

    labels = ("DRL-Agent", "FIFO", "SPT", "SPRPT")
    ax.legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    plt.show()