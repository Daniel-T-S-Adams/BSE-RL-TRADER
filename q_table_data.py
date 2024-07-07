import numpy as np 
import random
from typing import List, Dict, DefaultDict, Tuple
from collections import defaultdict
from tqdm import tqdm
import csv
from matplotlib import pyplot as plt

def load_q_table(file_path: str) -> DefaultDict:
    """
    Takes in a Q-table as a csv file and returns this information
    as a dictionary that is indexed by each state-action pair.

    :param file_path (str): The path to the file where the Q-table can be found.
    """
    q_table = defaultdict(lambda: 0)

    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                state, action, q_value = row
                q_table[(state, int(float(action)))] = float(q_value)

    except FileNotFoundError:
        pass  # If the file does not exist, return an empty q_table

    return q_table


def dump_q_table(q_table: DefaultDict, file_path: str):
    """
    Save the Q-table to a CSV file, updating existing entries or adding new ones.

    :param q_table (DefaultDict): The Q-table to save.
    :param file_path (str): The path to the file where the Q-table will be saved.
    """
    # Load the existing Q-table from the file
    existing_q_table = load_q_table(file_path)

    # Update existing Q-table with new entries
    for (state, action), q_value in q_table.items():
        existing_q_table[(state, action)] = q_value

    # Write the updated Q-table back to the file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['State', 'Action', 'Q-Value'])  # Write the header
        for (state, action), q_value in existing_q_table.items():
            writer.writerow([state, action, q_value])