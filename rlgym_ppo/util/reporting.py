"""
File: reporting.py
Author: Matthew Allen

Description:
    Misc. functions to log a dictionary of metrics to wandb and print them to the console.
"""


import torch
import numpy as np

def _form_printable_groups(report):
    """
    Function to create a list of dictionaries containing the data to print to the console in a specific order.
    :param report: A dictionary containing all of the values to organize.
    :return: A list of dictionaries containing keys organized in the desired fashion.
    """

    groups = [
        {"Policy Reward": report["Policy Reward"],
         "Policy Entropy": report["Policy Entropy"],
         "Value Function Loss": report["Value Function Loss"]},

        {"Mean KL Divergence": report["Mean KL Divergence"],
         "SB3 Clip Fraction": report["SB3 Clip Fraction"],
         "Policy Update Magnitude": report["Policy Update Magnitude"],
         "Value Function Update Magnitude": report["Value Function Update Magnitude"]},

        {"Collected Steps per Second": report["Collected Steps per Second"],
         "Overall Steps per Second": report["Overall Steps per Second"]},

        {"Timestep Collection Time": report["Timestep Collection Time"],
         "Timestep Consumption Time": report["Timestep Consumption Time"],
         "PPO Batch Consumption Time": report["PPO Batch Consumption Time"],
         "Total Iteration Time": report["Total Iteration Time"]},

        {"Cumulative Model Updates": report["Cumulative Model Updates"],
         "Cumulative Timesteps": report["Cumulative Timesteps"]},

        {"Timesteps Collected": report["Timesteps Collected"],
         "PPO Iterations": report["PPO Iterations"]},
              ]

    return groups

def report_metrics(loggable_metrics, debug_metrics, wandb_run=None):
    """
    Function to report a dictionary of metrics to the console and wandb.
    :param loggable_metrics: Dictionary containing all the data to be logged.
    :param debug_metrics: Optional dictionary containing extra data to be printed to the console for debugging.
    :param wandb_run: Wandb run to log to.
    :return: None.
    """

    if wandb_run is not None:
        wandb_run.log(loggable_metrics)

    # Print debug data first.
    if debug_metrics is not None:
        print("\nBEGIN DEBUG\n")
        print(dump_dict_to_debug_string(debug_metrics))
        print("\nEND DEBUG\n")


    # Print the loggable metrics in a desirable format to the console.
    print("{}{}{}".format("-"*8, "BEGIN ITERATION REPORT", "-"*8))
    groups = _form_printable_groups(loggable_metrics)
    out = ""
    for group in groups:
        out += dump_dict_to_debug_string(group) + "\n"
    print(out[:-2])
    print("{}{}{}\n\n".format("-"*8, "END ITERATION REPORT", "-"*8))

def dump_dict_to_debug_string(dictionary):
    """
    Function to format the data in a loggable dictionary so the line length is limited.

    :param dictionary: Data to format.
    :return: A string containing the formatted elements of that dictionary.
    """

    debug_string = ""
    for key, val in dictionary.items():
        if type(val) == torch.Tensor:
            if len(val.shape) == 0:
                val = val.detach().cpu().item()
            else:
                val = val.detach().cpu().tolist()

        # Format lists of numbers as [num_1, num_2, num_3] where num_n is clipped at 5 decimal places.
        if type(val) in (tuple, list, np.ndarray, np.array):
            arr_str = []
            for arg in val:
                arr_str.append("{:7.5f},".format(arg) if type(arg) == float else "{},".format(arg))
            arr_str = ' '.join(arr_str)
            debug_string = "{}{}: [{}]\n".format(debug_string, key, arr_str[:-1])

        # Format floats such that only 5 decimal places are shown.
        elif type(val) in (float, np.float32, np.float64):
            debug_string = "{}{}: {:7.5f}\n".format(debug_string, key, val)

        # Default to just printing the value if it isn't a type we know how to format.
        else:
            debug_string = "{}{}: {}\n".format(debug_string, key, val)

    return debug_string