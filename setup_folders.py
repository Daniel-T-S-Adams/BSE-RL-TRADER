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

    # Specify the file name
    file_name = "q_table_seller.csv"

    # Check if the file exists before attempting to delete it
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"{file_name} has been deleted.")
    else:
        print(f"{file_name} does not exist.")
        
    # Specify the file name
    file_name = "episode_seller.csv"

    # Check if the file exists before attempting to delete it
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"{file_name} has been deleted.")
    else:
        print(f"{file_name} does not exist.")
