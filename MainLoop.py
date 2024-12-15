from OrderIntake.OrderManager import OrderIntake, RandomOrder
from Simulation.S8Environment import Environment
from Agent.PPOAgent import Agent
import os
import random
import pythoncom
import traceback
import itertools
import pandas as pd


def save_results(results, memory_path):
    n_rows = range(len(results))
    df_expanded = pd.DataFrame(index=n_rows)
    df = pd.read_excel(memory_path)
    for col in df.columns:
        df_expanded[col] = pd.Series(df[col]).reindex(n_rows)
    df_expanded[f"Episode {ep}"] = results
    df_expanded.to_excel(memory_path, index=False)

    del df_expanded
    del df

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


if __name__=="__main__":

    print("--> Instantiate Environment")
    env = Environment()
    path = os.path.join(os.getcwd(), "Simulation", "JobShop.S8")
    print("--> Open Sim")
    env.sim.open(path)

    print("--> Instantiate Agents")
    number_of_selectable_heuristics = 4
    actor_model_1 = os.path.join(os.getcwd(), "Agent", "Model_Stage_1", "Run 9", "ActorEP3.pth")
    critic_model_1 = os.path.join(os.getcwd(), "Agent", "Model_Stage_1", "Run 9", "CriticEP3.pth")
    agent_1 = Agent(n_actions=number_of_selectable_heuristics, n_states=6, alpha=0.0002, beta=0.0005, gamma=0.99,
                    lambd=0.95, clip=0.2, batch_size=256, actor_model=actor_model_1, critic_model=critic_model_1)
    number_of_selectable_orders = 5
    actor_model_2 = os.path.join(os.getcwd(), "Agent", "Model_Stage_2", "Run 9", "ActorEP3.pth")
    critic_model_2 = os.path.join(os.getcwd(), "Agent", "Model_Stage_2", "Run 9", "CriticEP3.pth")
    agent_2 = Agent(n_actions=number_of_selectable_orders, n_states=number_of_selectable_orders * 5, alpha=0.0002,
                    beta=0.0005, gamma=0.99, lambd=0.95, clip=0.2, batch_size=256, actor_model=actor_model_2,
                    critic_model=critic_model_2)

    print("--> Instantiate Order Intake")
    oi = OrderIntake()

    sequence_length = 1024
    training_episodes = 10
    n_default_orders = 15

    for ep in itertools.count(start=1):
        print(f"\n\n---------------------------- Episode {ep} ----------------------------")
        avg_slack_time_history =  []
        avg_machine_utilization_history = []
        reward_history = []

        n_training_runs = 0
        done = False
        training_done = False

        new_decision = False
        current_state_1 = None
        current_state_2 = None
        action_1 = None
        action_2 = None
        log_prob_1 = None
        log_prob_2 = None
        reward = None
        state_value_1 = None
        state_value_2 = None

        print("--> Clear Order Intake")
        oi.clear()
        print("--> Clear Production Lines")
        env.problem.clear_all_queues()
        print("--> Clear Sim State")
        env.problem.clear_sim_state()
        print("--> Problem to Excel")
        env.problem.problem_to_excel()

        print("\n--> Generate Initial Orders")
        for _ in range(n_default_orders):
            new_order = RandomOrder()
            data = new_order.get_data()
            if not oi.duplicat(data["OrderNumber"]):
                oi.append(data)
        print("--> Insert new orders to Queues")
        env.problem.oi_to_queues(oi)

        print("--> Set the starting Orders to the simulation queues")
        for queue in env.problem.queues:
            of_sys = queue.of_system
            if env.problem.order_in_queue(of_sys):
                initial_order = env.problem.get_random_order_from_queue(of_sys)
                env.problem.order_to_state(initial_order)
                env.problem.set_label_data(of_system=of_sys, order=initial_order)
                env.sim.wi_to_queue(nb_wi=len(initial_order), queue=of_sys)

        env.problem.problem_to_excel()
        env.sim.excel_to_sim()
        print("--> Start Simulation")
        env.sim.run(0, sim_speed=100)

        env.sim.listenForMessages = True
        try:
            # Start loop to listen for events
            while env.sim.listenForMessages and not done:
                # Process messages and return as soon as there are no more waiting messages to process
                pythoncom.PumpWaitingMessages()

                if env.sim.new_request:
                    print(f"\n\n-----------Episode: {ep}, Performed steps: {len(agent_1.sequence)}/{sequence_length}-----------")
                    print("\n--> New Request")
                    env.sim.sim_to_excel()
                    env.problem.excel_to_problem()
                    env.problem.update_queues()

                    requests = env.sim.get_request()

                    for request in requests:
                        command = request["Command"]
                        attributes = request["Attributes"]
                        # Command: MoveOn, Attributes: FromSystem, ToSystem, OrderNumber
                        if command == "MoveOn":
                            from_system = attributes[0]
                            to_system = attributes[1]
                            order_number = attributes[2]
                            print(f"\n--> Request: \n    Command: {command}, FromSystem: {from_system}, ToSystem: {to_system}, OrderNumber: {order_number}")
                            env.problem.move_to_next_queue(f"{order_number}", f"{from_system}", f"{to_system}")
                        # Command: DeleteOrder, Attributes: FromQueue, OrderNumber
                        elif command == "DeleteOrder":
                            from_system = attributes[0]
                            order_number = attributes[1]
                            print(f"\n--> Request: \n    Command: {command}, FromSystem: {from_system}, OrderNumber: {order_number}")
                            env.problem.remove_order_from_queue(f"{from_system}", f"{order_number}")
                            env.problem.remove_order_from_state(f"{order_number}")
                        # Command: NewOrder, Attributes: ToSystem
                        elif command == "NewOrder":
                            to_system = attributes[0]
                            print(f"\n--> Request: \n    Command: {command}, ToSystem: {to_system}")

                            if new_decision:
                                reward = env.problem.compute_reward()
                                print(f"    Reward: {reward}")
                                reward_history.append(reward)

                                agent_1.sequence.set(current_state_1,
                                                     action_1,
                                                     reward,
                                                     log_prob_1,
                                                     state_value_1,
                                                     False)
                                agent_2.sequence.set(current_state_2,
                                                     action_2,
                                                     reward,
                                                     log_prob_2,
                                                     state_value_2,
                                                     False)

                                avg_machine_utilization = env.problem.get_avg_machine_utilization()
                                print(f"    Avg. Machine Utilization: {avg_machine_utilization}")
                                avg_machine_utilization_history.append(avg_machine_utilization)

                                avg_slack_time = env.problem.get_avg_slack_time()
                                print(f"    Avg. Slack Time: {avg_slack_time}")
                                avg_slack_time_history.append(avg_slack_time)

                            # Randomly generate new orders
                            if not to_system == "FD" and not env.problem.n_orders_in_queue(of_system=to_system) > 30:
                                if random.randint(0, 1) or not env.problem.order_in_queue(of_system=to_system):
                                    print("\n!!!!!!!New orders arrived!!!!!!!!!")
                                    for _ in range(random.randrange(7, 10)):
                                        new_order = RandomOrder()
                                        data = new_order.get_data()
                                        if not oi.duplicat(data["OrderNumber"]):
                                            oi.append(data)
                                    print("--> Insert new orders to Queue of production line\n")
                                    env.problem.oi_to_queues(oi)

                            if len(agent_1.sequence) == sequence_length:
                                print(f"\n!!!!!!!Perform Training!!!!!!!")
                                agent_1.perform_training(training_episodes)
                                agent_2.perform_training(training_episodes, masked=True)
                                training_done = True

                            if env.problem.order_in_queue(to_system):
                                new_decision = True
                                current_state_1 = env.problem.get_avg_state(to_system)
                                # print(f"    Current State 1: {current_state_1}")
                                action_1, log_prob_1 = agent_1.get_action(current_state_1)
                                # print(f"    Action 1: {action_1}")
                                heuristic = env.problem.heuristics[action_1]
                                print(f"    Heuristic: {heuristic}")
                                env.problem.sort_queue(to_system, heuristic)
                                state_value_1 = agent_1.get_state_value(current_state_1)
                                # print(f"    State Value 1: {state_value_1}")

                                current_state_2, lst_order_numbers = env.problem.get_order_specific_state(to_system, of_n_orders=number_of_selectable_orders)
                                # print(f"    Current State 2: {current_state_2}")
                                action_2, log_prob_2 = agent_2.get_action(current_state_2, masked=True)
                                # print(f"    Action 2: {action_2}, LogProb 2: {log_prob_2}")
                                state_value_2 = agent_2.get_state_value(current_state_2)
                                # print(f"    State Value 2: {state_value_2}")

                                nb_chosen_order = lst_order_numbers[action_2]
                                print(f"    Chosen Order: {nb_chosen_order}")
                                chosen_order = env.problem.get_order_from_queue(to_system, nb_chosen_order)
                                if to_system == "FD":
                                    env.problem.update_state(chosen_order)
                                    sys = chosen_order.at[0, "ItemType"]
                                    env.sim.link_sim_objects(f"{sys}")
                                else:
                                    env.problem.order_to_state(chosen_order)
                                    env.problem.set_label_data(of_system=to_system, order=chosen_order)
                                    env.sim.wi_to_queue(nb_wi=len(chosen_order), queue=to_system)
                            else:
                                new_decision = False

                    if training_done:
                        done = True
                        env.sim.new_request = False
                    else:
                        env.problem.problem_to_excel()
                        env.sim.excel_to_sim()
                        env.sim.new_request = False
                        env.sim.run(0, sim_speed=100)

        except Exception:
            print(traceback.format_exc())
            break


        file_directory = os.path.dirname(__file__)
        slack_time_results_directory = os.path.join(file_directory, "Results", "Training", f"AvgSlackTime.xlsx")
        save_results(avg_slack_time_history, slack_time_results_directory)
        machine_utilization_results_directory = os.path.join(file_directory, "Results", "Training", f"AvgMachineUtilization.xlsx")
        save_results(avg_machine_utilization_history, machine_utilization_results_directory)
        rewards_directory = os.path.join(file_directory, "Results", "Training", f"Rewards.xlsx")
        save_results(reward_history, rewards_directory)

        print("--> Save Model")
        agent_1.save_model(stage=1, episode=ep)
        agent_2.save_model(stage=2, episode=ep)
        print("--> Reset Simulation")
        env.sim.reset()

    print("\n\n--> Close Simulation")
    env.sim.close()
