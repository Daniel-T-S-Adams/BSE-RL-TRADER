# The file that is used to run the code.

import logging

# Importing Global Parameters 
from config.config_params import CONFIG


## Tabular or FA imports : ##

if CONFIG['function_approximation']: 
    # Import the main function for training
    from FA_train_trader import train
    
elif CONFIG['tabular']:
     
    # Import the main function for training
    from tab_train_trader import train



# Import the main function for testing
from test_Policies import Test_all_policies

## Shared Imports : ## 
# Import function that creates subfolders
from setup_folders import create_subfolders
# Import function that cleans up the directory
from setup_folders import delete_files
# Import the main function for plotting
from Plotting import create_plots

# Configure logging in the main script
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


def main():
    """
    Main function to set up subfolders, train the RL agent, test policies, 
    plot results, and clean up the directory.

    Steps:
    1. Creates necessary subfolders.
    2. Trains the reinforcement learning agent with specified parameters.
    3. Tests all policies after training.
    4. Creates plots based on saved statistics.
    5. Cleans up files in the directory for the next run.
    """

    # Sets up the subfolders
    create_subfolders()



    # Training the RL agent with testing
    print("Started the Training")
    train(
        total_eps=CONFIG['total_eps'], 
        market_params=CONFIG['market_params'], 
        epsilon_start=CONFIG['epsilon_start']
    )
    print("Finished the Training")

    # Testing all policies
    print("Started the Testing")
    saved_stats = Test_all_policies(
        CONFIG['GPI_test_freq'], 
        CONFIG['num_GPI_iter'], 
        CONFIG['market_params']
    )
    print("Finished the Testing")

    # Do all the plotting
    create_plots(saved_stats)

    # Clean up the directory by deleting files for the next run
    delete_files()

if __name__ == "__main__":
    main()