# The file that is used to run the code.


# Importing Global Parameters
from GlobalParameters import CONFIG

# Import function that creates subfolders
from setup_folders import create_subfolders
# Import function that cleans up the directory
from setup_folders import delete_files


# Import the main function for training
from train_trader import train

# Import the main function for testing
from Test_Policies import Test_all_policies

# Import the main function for plotting
from Plotting import create_plots


# Sets up the subfolders
create_subfolders()

# Training the RL agent with testing
train( total_eps=CONFIG['total_eps'], 
        market_params=CONFIG['market_params'], 
        test_freq=CONFIG['test_freq'],
        epsilon_start=CONFIG['epsilon'])


print(f"Finished the Training")


saved_stats = Test_all_policies(CONFIG['GPI_test_freq'], CONFIG['num_GPI_iter'], CONFIG['market_params'],CONFIG['epsilon'])

print(f"Finished the Testing")


# Do all the plotting
create_plots(saved_stats)


# Clean up the directory by deleting files for the next run
delete_files()


