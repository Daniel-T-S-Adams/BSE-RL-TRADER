# tabular RL, folder handling 


import os
from config.config_params import CONFIG

def create_subfolders():
    
    ## Shared files : ##
    if not os.path.exists(CONFIG["plots"]):
        os.makedirs(CONFIG["plots"])
    # Path to the source file (config_params.py)
    source_file_path = os.path.join(os.path.dirname(__file__), 'config', 'config_params.py')

    # Ensure the source file exists
    if not os.path.exists(source_file_path):
        raise FileNotFoundError(f"Source file not found: {source_file_path}")
    
    # Destination path specified in CONFIG['parameters']
    destination_file_path = CONFIG['parameters']

    # Copy the content of source_file_path into destination_file_path
    with open(source_file_path, 'r') as source_file:
        with open(destination_file_path, 'w') as destination_file:
            destination_file.write(source_file.read())

    ## Files for the tabular method : ##
        if CONFIG['tabular']:
            if not os.path.exists(CONFIG["q_tables"]):
                os.makedirs(CONFIG["q_tables"])
            
            if not os.path.exists(CONFIG["counts"]):
                os.makedirs(CONFIG["counts"])
                
    
    ## Files for the FA method : ## 
        elif CONFIG['function_approximation']:
            if not os.path.exists(CONFIG["weights"]):
                os.makedirs(CONFIG["weights"])
                
    
    print("Subfolders created or already exist.")


def delete_files():
    ### Delete folders for next run. 

    # List of files to delete
    files_to_delete = [
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