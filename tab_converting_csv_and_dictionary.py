from typing import Dict, DefaultDict
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
        print(f"No File found at : {file_path}, returning empty q_table")
        pass  # If the file does not exist, return an empty q_table

    return q_table

            
            
# Function that takes a q_table as a dictionary and saves a CSV file.
def save_q_table_dict_to_csv(new_q_table: Dict, filename : str):
    """
    Save the Q-table to a CSV file.

    Parameters:
        new_q_table (Dict): The Q-table to save.
    """
    sorted_new_q_table = sorted(new_q_table.items(), key=lambda x: x[0][0])
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['State', 'Action', 'Q-Value'])
        # Write the data
        for (state, action), q_value in sorted_new_q_table:
            writer.writerow([state, action, q_value])
            
            
            
            
# Function to save state action counts dictionary to a CSV file
def save_sa_counts_to_csv(sa_counts, filename):
    """
    Save state-action counts to a CSV file.

    Parameters:
        sa_counts (dict): Dictionary containing state-action counts.
        filename (str): The filename to save the counts.
    """
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header
        writer.writerow(['State', 'Action', 'Count'])
        # Write the data
        for (state, action), count in sa_counts.items():
            writer.writerow([state, action, count])
