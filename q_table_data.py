from typing import List, Dict, DefaultDict, Tuple
from collections import defaultdict
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
                state_str, action, q_value = row
                state = tuple(map(float, state_str.strip('()').split(',')))
                q_table[(state, float(action))] = float(q_value)

    except FileNotFoundError:
        pass  # If the file does not exist, return an empty q_table

    return q_table


def update_q_table(q_table: DefaultDict, old_q_table: dict):
    """
    Update the q_table with entries from old_q_table that are not already in q_table.

    :param q_table (DefaultDict): The Q-table to update.
    :param old_q_table (dict): The old Q-table with potential additional entries.
    """
    for key, value in old_q_table.items():
        if key not in q_table:
            q_table[key] = value

def dump_q_table(q_table: DefaultDict, file_path: str):
    """
    Save the Q-table to a CSV file.

    :param q_table (DefaultDict): The Q-table to save.
    :param file_path (str): The path to the file where the Q-table will be saved.
    """
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['State', 'Action', 'Q-Value'])  # Write the header
        for (state, action), q_value in q_table.items():
            state_str = f"({','.join(map(str, state))})"  # Convert state tuple to string
            writer.writerow([state_str, action, q_value])
            
            
# # Example usage
# # our q-table
# q_table = defaultdict(lambda: 0, {((1.0, 2.0), 3.0): 0.6, ((4.0, 5.0), 6.0): 0.7})
# # save it 
# dump_q_table(q_table, 'q_table_seller_episode_num3.csv')
# # load it (this is where it starts)
# q_table = load_q_table('q_table_seller_episode_num3.csv')
# # edit it 
# q_table[((1.0, 2.0), 3.0)] = 0.5
# # save it again
# dump_q_table(q_table, 'q_table_seller_episode_num3.csv')
            
# Example usage
# old_q_table = defaultdict(lambda: 0, {((1.0, 2.0), 3.0): 0.7, ((4.0, 5.0), 6.0): 0.7})
# q_table = defaultdict(lambda: 0, {((1.0, 2.0), 3.0): 0.5, ((1.0, 1.0), 1.0): 0.1})
# update_q_table(q_table, old_q_table)
# dump_q_table(q_table, 'q_table_seller.csv')