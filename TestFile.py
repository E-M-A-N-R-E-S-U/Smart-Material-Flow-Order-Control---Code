import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

# col = ["OrderNumber", "ItemType"]
#
# a = pd.DataFrame({"OrderNumber": ("12345", "12345", "678910", "1112131415"), "ItemType": ("1", "1", "2", "3")}, columns=col)
# print(a)
# b = pd.DataFrame({"OrderNumber": ("12345", "678910"), "ItemType": ("1", "2")}, columns=col)
# b = b.drop_duplicates()
#
# mask = a["OrderNumber"] == "12345"
# d = a[~mask]
# print(d)

# if len(a["OrderNumber"]) != len(b["OrderNumber"]):
#     df_all = a.merge(b, on=["OrderNumber"], how="left", indicator=True)
#     print(df_all)
#     df_only = df_all[df_all["_merge"] == "left_only"]
#     print(df_only.drop(["_merge"], axis=1))
#

# def sim_time_to_date(value):
#     if not type(value) == float:
#         value = float(value)
#
#     windows_start_date = datetime(1900, 1, 1)
#
#     end_date = windows_start_date + timedelta(days=value)
#
#     return end_date
#
# # print(sim_time_to_date(45545.375))
#
# operation_times = {"S": 10, "M": 10, "MF": 20, "E": 25, "EH": 20, "VB": 15, "SWD": 30, "VSW": 10,
#                    "FD": 10, "VFD": 15}
# template = r"C:\Users\wm01710\Documents\Projekte\Automatisierung Materialflussplanung\03 Realisierung\Main\Simulation\Queues\Test_Template.xlsx"
# template_df = pd.read_excel(template)
#
# def time_remaining_2(order: pd.DataFrame):
#     order_specific_process_times = []
#     t_s = order.at[0, "S"] * operation_times["S"]
#     order_specific_process_times.append(t_s)
#     t_m = order.at[0, "M"] * operation_times["M"]
#     order_specific_process_times.append(t_m)
#     t_mf = order.at[0, "MF"] * operation_times["MF"]
#     order_specific_process_times.append(t_mf)
#     t_e = order.at[0, "E"] * operation_times["E"]
#     order_specific_process_times.append(t_e)
#     t_eh = order.at[0, "EH"] * operation_times["EH"]
#     order_specific_process_times.append(t_eh)
#     t_vb = order.at[0, "VB"] * operation_times["VB"]
#     order_specific_process_times.append(t_vb)
#     t_swd = order.at[0, "SWD"] * operation_times["SWD"]
#     order_specific_process_times.append(t_swd)
#     t_vsw = order.at[0, "VSW"] * operation_times["VSW"]
#     order_specific_process_times.append(t_vsw)
#     t_fd = order.at[0, "FD"] * operation_times["FD"]
#     order_specific_process_times.append(t_fd)
#     t_vfd = order.at[0, "VFD"] * operation_times["VFD"]
#     order_specific_process_times.append(t_vfd)
#
#     min_process_time_to_complete_one_component = sum(order_specific_process_times)
#     longest_process_time_on_one_machine = max(order_specific_process_times)
#
#     finished_components = 0
#     t_remaining = 0
#     for i in range(len(order)):
#         t_remaining = 0
#         t_remaining += order.at[i, "S"] * abs(order.at[i, "S_Done"] - 1) * operation_times["S"]
#         t_remaining += order.at[i, "M"] * abs(order.at[i, "M_Done"] - 1) * operation_times["M"]
#         t_remaining += order.at[i, "MF"] * abs(order.at[i, "MF_Done"] - 1) * operation_times["MF"]
#         t_remaining += order.at[i, "E"] * abs(order.at[i, "E_Done"] - 1) * operation_times["E"]
#         t_remaining += order.at[i, "EH"] * abs(order.at[i, "EH_Done"] - 1) * operation_times["EH"]
#         t_remaining += order.at[i, "VB"] * abs(order.at[i, "VB_Done"] - 1) * operation_times["VB"]
#         t_remaining += order.at[i, "SWD"] * abs(order.at[i, "SWD_Done"] - 1) * operation_times["SWD"]
#         t_remaining += order.at[i, "VSW"] * abs(order.at[i, "VSW_Done"] - 1) * operation_times["VSW"]
#         t_remaining += order.at[i, "FD"] * abs(order.at[i, "FD_Done"] - 1) * operation_times["FD"]
#         t_remaining += order.at[i, "VFD"] * abs(order.at[i, "VFD_Done"] - 1) * operation_times["VFD"]
#         if t_remaining == 0:
#             finished_components += 1
#         else:
#             break
#
#     if finished_components == 0:
#         process_time_so_far = min_process_time_to_complete_one_component - t_remaining
#     else:
#         process_time_so_far_of_last_completed_component = min_process_time_to_complete_one_component + (finished_components - 1) * longest_process_time_on_one_machine
#         process_time_so_far_of_last_uncompleted_component = min_process_time_to_complete_one_component - t_remaining + finished_components * longest_process_time_on_one_machine
#         if process_time_so_far_of_last_completed_component == process_time_so_far_of_last_uncompleted_component:
#             process_time_so_far = process_time_so_far_of_last_completed_component
#         else:
#             process_time_so_far = process_time_so_far_of_last_completed_component + (process_time_so_far_of_last_uncompleted_component - process_time_so_far_of_last_completed_component)
#
#     total_processing_time_of_order = min_process_time_to_complete_one_component + (len(order) - 1) * longest_process_time_on_one_machine
#     total_time_remaining = total_processing_time_of_order - process_time_so_far
#
#     return total_time_remaining
#
#
# def time_remaining(order: pd.DataFrame):
#     processing_times = []
#     t_s = order.at[0, "S"] * operation_times["S"]
#     processing_times.append(t_s)
#     t_m = order.at[0, "M"] * operation_times["M"]
#     processing_times.append(t_m)
#     t_mf = order.at[0, "MF"] * operation_times["MF"]
#     processing_times.append(t_mf)
#     t_e = order.at[0, "E"] * operation_times["E"]
#     processing_times.append(t_e)
#     t_eh = order.at[0, "EH"] * operation_times["EH"]
#     processing_times.append(t_eh)
#     t_vb = order.at[0, "VB"] * operation_times["VB"]
#     processing_times.append(t_vb)
#     t_swd = order.at[0, "SWD"] * operation_times["SWD"]
#     processing_times.append(t_swd)
#     t_vsw = order.at[0, "VSW"] * operation_times["VSW"]
#     processing_times.append(t_vsw)
#     t_fd = order.at[0, "FD"] * operation_times["FD"]
#     processing_times.append(t_fd)
#     t_vfd = order.at[0, "VFD"] * operation_times["VFD"]
#     processing_times.append(t_vfd)
#
#     # Der späteste Fertigstellungszeitpunkt wird nur berechnet, wenn das erste Bauteil des Auftrags (Index 0) darauf
#     # wartet weitergeleitet zu werden oder wenn noch gar kein Bauteil des Auftrags auf irgendeiner Anlage des Systems
#     # ist. Außerdem kann immer nur ein Auftrag gleichzeitig auf einer Anlage sein. Es gibt also nie den Fall, dass der
#     # späteste Fertigstellungszeitpunkt berechnet wird, wenn mehrere Bauteile des Auftrags vollständig bearbeitet sind.
#     # Daher kann die bisherige Bearbeitungszeit des Auftrags am ersten Bauteil des Auftrags abgelesen werden.
#     processing_time_so_far = []
#     t_s_done = order.at[0, "S_Done"] * t_s
#     processing_time_so_far.append(t_s_done)
#     t_m_done = order.at[0, "M_Done"] * t_m
#     processing_time_so_far.append(t_m_done)
#     t_mf_done = order.at[0, "MF_Done"] * t_mf
#     processing_time_so_far.append(t_mf_done)
#     t_e_done = order.at[0, "E_Done"] * t_e
#     processing_time_so_far.append(t_e_done)
#     t_eh_done = order.at[0, "EH_Done"] * t_eh
#     processing_time_so_far.append(t_eh_done)
#     t_vb_done = order.at[0, "VB_Done"] * t_vb
#     processing_time_so_far.append(t_vb_done)
#     t_swd_done = order.at[0, "SWD_Done"] * t_swd
#     processing_time_so_far.append(t_swd_done)
#     t_vsw_done = order.at[0, "VSW_Done"] * t_vsw
#     processing_time_so_far.append(t_vsw_done)
#     t_fd_done = order.at[0, "FD_Done"] * t_fd
#     processing_time_so_far.append(t_fd_done)
#     t_vfd_done = order.at[0, "VFD_Done"] * t_vfd
#     processing_time_so_far.append(t_vfd_done)
#
#     total_runtime = sum(processing_times)
#     total_processing_time_so_far = sum(processing_time_so_far)
#     t_bottleneck = max(processing_times)
#     tr = total_runtime + (len(order) - 1) * t_bottleneck
#     tr = tr - total_processing_time_so_far
#
#     return tr
#
# def waiting_time(order: pd.DataFrame):
#     wt = order.at[0, "t_step"] - order.at[0, "t_slot"]
#     return wt
#
# def slack_time(order: pd.DataFrame, time_remaining):
#     st = order.at[0, "Deadline"] - (order.at[0, "t_step"] + time_remaining)
#     return st
#
#
# if __name__=="__main__":
#     included_orders = template_df.drop_duplicates(subset=["OrderNumber"])
#     order_numbers = included_orders["OrderNumber"].to_list()
#     sum_time_remaining = 0
#     sum_time_remaining_2 = 0
#     sum_waiting_time = 0
#     sum_slack_time = 0
#     for order_number in order_numbers:
#         order = template_df[template_df.OrderNumber.isin([order_number])].reset_index(drop=True)
#         tr = time_remaining(order)
#         sum_time_remaining_2 += time_remaining_2(order)
#         sum_time_remaining += tr
#         sum_waiting_time += waiting_time(order)
#         sum_slack_time += slack_time(order, tr)
#     avg_time_remaining_2 = sum_time_remaining_2 / len(order_numbers)
#     avg_time_remaining = sum_time_remaining / len(order_numbers)
#     avg_waiting_time = sum_waiting_time / len(order_numbers)
#     avg_slack_time = sum_slack_time / len(order_numbers)
#     print(round(avg_time_remaining_2, 3))
#     print(round(avg_time_remaining, 3))
#     print(round(avg_waiting_time, 3))
#     print(round(avg_slack_time, 3))

# reward_history_1 = [{f"Episode 0": [10, 20, 30, 20, 22, 33, 45, 76, 55]},
#                   {f"Episode 1": [13, 25, 36, 34, 40, 38]},
#                   {f"Episode 2": [44, 34, 56, 60]}]
# reward_history_2 = [{f"Episode 0": [44, 34, 56, 60, 67, 79, 83, 85, 95]},
#                     {f"Episode 1": [13, 25, 36, 34, 40, 38, 48, 66, 79]},
#                     {f"Episode 2": [10, 20, 30, 20, 22, 33, 45, 76, 100]}]
# plt.figure(1)
# for idx, reward_history_of_episode in enumerate(reward_history_1):
#     x = [i + 1 for i in range(len(reward_history_of_episode[f"Episode {idx}"]))]
#     plt.plot(x, reward_history_of_episode[f"Episode {idx}"], label=f"Episode {idx}")
# plt.title("Avg. Slack Time")
# plt.legend()
# plt.savefig("Results/Avg Slack Time.png")
#
# plt.figure(2)
# for idx, reward_history_of_episode in enumerate(reward_history_2):
#     x = [i + 1 for i in range(len(reward_history_of_episode[f"Episode {idx}"]))]
#     plt.plot(x, reward_history_of_episode[f"Episode {idx}"], label=f"Episode {idx}")
# plt.title("Avg. Utilization")
# plt.legend()
# plt.savefig("Results/Avg Utilization.png")
# plt.show()

# data1 = {"OrderNumber": [1,2,3,4,5],
#          "NewEntry": ["False", "False", "False", "True", "True"]}
# df1 = pd.DataFrame(data1)
# print(df1)
#
# data2 = {"OrderNumber": [4, 5],
#          "NewEntry": ["False", "False"]}
# df2 = pd.DataFrame(data2)
# print(df2)
#
# on = df2["OrderNumber"].to_list()
# for o in on:
#     df1.loc[df1["OrderNumber"] == o, "NewEntry"] = "False"
#
# print(df1)

# columns = ["Col1", "Col2", "Col3"]
# df = pd.DataFrame(columns=columns)
#
# print(df["Col1"].tolist())

columns = {"OrderNumber": [1, 2, 3, 4, 5],
           "InQueue": [1, 1, 1]}

max_len = 0
for arr in columns.values():
    if len(arr) > max_len:
        max_len = len(arr)

for key in columns.keys():
    diff = max_len - len(columns[key])
    if diff > 0:
        for i in range(diff):
            columns[key].append(0)

df = pd.DataFrame(columns)
print(df)
if (df["InQueue"] == "FD").all():
    print("True")