# A file that sets up the subfolders

import os
from GlobalParameters import CONFIG

def create_subfolders():
    if not os.path.exists(CONFIG["q_tables"]):
        os.makedirs(CONFIG["q_tables"])
    
    if not os.path.exists(CONFIG["counts"]):
        os.makedirs(CONFIG["counts"])
        
    if not os.path.exists(CONFIG["plots"]):
        os.makedirs(CONFIG["plots"])
        
    # Get the path to GlobalParameters.py
    global_parameters_file = 'GlobalParameters.py'

    # Copy the content of GlobalParameters.py into the new file specified by CONFIG['parameters']
    with open(global_parameters_file, 'r') as source_file:
        with open(CONFIG['parameters'], 'w') as destination_file:
            destination_file.write(source_file.read())

    print("Subfolders created or already exist.")


def delete_files():
    ### Delete folders for next run. 

    # List of files to delete
    files_to_delete = [
    "q_table_seller.csv",
    "episode_seller.csv",
    "session_1_avg_balance.csv",
    "session_1_blotters.csv",
    "session_1_LOB_frames.csv",
    "session_1_strats.csv",
    "session_1_tape.csv"]

    # Delete each file
    for file_name in files_to_delete:
        try:
            if os.path.exists(file_name):
                os.remove(file_name)
                print(f"Deleted {file_name}")
            else:
                print(f"{file_name} does not exist")
        except Exception as e:
            print(f"Error deleting {file_name}: {e}")