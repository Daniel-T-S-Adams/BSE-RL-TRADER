# A file for testing the performance of the trader.
# The main function takes a q_table and a number of episodes to test that policy for.

# Imports from Standard Library
import logging
logger = logging.getLogger(__name__)



# Imports from Third Party Modules
from BSE import market_session
from GlobalParameters import CONFIG
from Plotting import create_plots
from epsilon_scheduling import linear_epsilon_decay
from q_table_data import load_q_table

def test_policy(episodes: int, market_params: tuple) -> dict:
    """
    Test a trading policy over a specified number of episodes.

    Parameters:
        episodes (int): Number of episodes to test the policy.
        market_params (tuple): Parameters for the market session.

    Returns:
        dict: Cumulative average profit statistics for each trader type.
    """
    updated_market_params = list(market_params)

    # Initialize an empty dictionary to store cumulative average profit
    cumulative_stats = {}
    # For storing previous profit (unused currently)
    previous_avg_profit = None

    for episode in range(episodes):
        # Run the market session
        market_session(*updated_market_params)

        # Read the average profit file at the final timestep of each market session
        # Average here is over all traders of a given type
        current_stats = read_average_profit('session_1_avg_balance.csv')

        # Update cumulative average profit
        update_cumulative_average_profit(cumulative_stats, current_stats)

    # Calculate average profit for each trader type over all episodes
    for ttype in cumulative_stats:
        cumulative_stats[ttype]['avg_profit'] /= episodes

    return cumulative_stats

def update_cumulative_average_profit(cumulative_stats, new_stats):
    """
    Update cumulative average profit statistics with new episode data.

    Parameters:
        cumulative_stats (dict): Cumulative statistics to update.
        new_stats (dict): New statistics from the current episode.
    """
    for ttype, stats in new_stats.items():
        if ttype in cumulative_stats:
            cumulative_stats[ttype]['avg_profit'] += stats['avg_profit']
        else:
            cumulative_stats[ttype] = {
                'total_profit': stats['total_profit'],
                'num_traders': stats['num_traders'],
                'avg_profit': stats['avg_profit']
            }

def read_average_profit(file_path):
    """
    Read average profit data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        dict: Trader statistics extracted from the file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Get the final line
    final_line = lines[-1]

    # Split the final line into components
    components = final_line.strip().split(', ')

    # Initialize an empty dictionary to store trader stats
    trader_stats = {}
    # Skip the first four components (sesid, time, best bid, best offer)
    index = 4
    # Iterate over the components to extract trader stats
    while index < len(components):
        ttype = components[index]
        total_profit = float(components[index + 1])
        num_traders = int(components[index + 2])
        # Clean up the string before converting to float by removing commas
        avg_profit = float(components[index + 3].replace(',', ''))
        trader_stats[ttype] = {
            'total_profit': total_profit,
            'num_traders': num_traders,
            'avg_profit': avg_profit
        }
        index += 4

    return trader_stats

def Test_all_policies(GPI_test_freq: int, num_GPI_iters: int, market_params: tuple) -> list:
    """
    Test the performance of policies after specified GPI iterations.

    Parameters:
        GPI_test_freq (int): Frequency of GPI iterations to test.
        num_GPI_iters (int): Total number of GPI iterations.
        market_params (tuple): Parameters for the market session.

    Returns:
        list: A list of dictionaries containing cumulative stats for each tested GPI iteration.
    """
    # Test the performance for the specified GPI iterations
    iters_to_test = list(range(1, num_GPI_iters + 1, GPI_test_freq))

    saved_stats = []
    for GPI_iter in iters_to_test:
        q_table_string = CONFIG['setup'] + f'\\q_tables\\q_table_seller_after_GPI_{GPI_iter}.csv'
        logger.info(f"Testing the performance after GPI iteration {GPI_iter}")
        logger.info(f"Using q_table: {q_table_string}")

        q_table = load_q_table(q_table_string)
        market_params[3]['sellers'][1][2]['q_table_seller'] = q_table
        market_params[3]['sellers'][1][2]['epsilon'] = 0.0
        

        cumulative_stats = test_policy(episodes=CONFIG['test_episodes'], market_params=market_params)

        # Log the performance statistics
        for ttype in cumulative_stats:
            logger.info(f"Performance Test: GPI Iter {GPI_iter}, {ttype} average profit: {cumulative_stats[ttype]['avg_profit']}")

        saved_stats.append(cumulative_stats)

    return saved_stats  # saved_stats is a list of dictionaries, one for each GPI iteration.


