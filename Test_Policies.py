# A file for testing the performance of the trader. 
# The main function takes a q_table and a number of epiodes to test that policy for.


from BSE import market_session
from GlobalParameters import CONFIG

# Import the main function for plotting
from Plotting import create_plots
from epsilon_scheduling import epsilon_decay

# The following function takes a q_table and a number of epiodes to test that policy for.
def test_policy(episodes: int, market_params: tuple) -> dict:

    updated_market_params = list(market_params)    

    # initialize an empty dictionary to store cumulative average profit
    cumulative_stats = {}
    # for storing previous profit
    previous_avg_profit = None
    
    for episode in range(episodes):
        # Run the market session
        market_session(*updated_market_params)
        
        # Read the average profit file at the final timestep of each market session. Average here is over all traders of a given type!!!
        current_stats = read_average_profit('session_1_avg_balance.csv')
        
        
        # getting a cumulative tally of the average profit for each trader type. Again Average here is over all traders of a given type!!! Cumulative just means 
        # we are adding this average profit for each episode to the previous episodes.
        update_cumulative_average_profit(cumulative_stats, current_stats)


    #     ##WORK IN PROGRESS.
    #     # CONVERGENCE CHECK. this needs to be changed to make sure each traders average profit has converged. at the moment we are just summing them.
    #     # Calculate average profit so far. Here by average we mean averaging over all episodes so far.
    #     current_avg_profit = sum([cumulative_stats[ttype]['avg_profit'] for ttype in cumulative_stats])/(episode+1)
    #     # Check for convergence every 100 steps
    #     if episode % 100 == 0:
    #         if previous_avg_profit is not None:
    #             profit_change = abs(current_avg_profit - previous_avg_profit)
    #             if profit_change <= 0.00005:
    #                 print(f"Convergence achieved at episode {episode} with profit change {profit_change}")
    #                 # get the average over all episodes
    #                 for ttype in cumulative_stats:
    #                     cumulative_stats[ttype]['avg_profit'] /= (episode+1)
    #                 return cumulative_stats
            
            
    #         previous_avg_profit = current_avg_profit
       
    # # get the average over all episodes if we dont converge
    # print(f"Did not converge after {episodes} episodes")
    # ####WORK IN PROGRESS.
    
    # Calculate average profit for each trader type. Here by average we mean averaging over all episodes.
    for ttype in cumulative_stats:
        cumulative_stats[ttype]['avg_profit'] /= episodes
        
       
    return cumulative_stats


def update_cumulative_average_profit(cumulative_stats, new_stats):
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


# The following function tests the performance of the policies for specified GPI iterations
def Test_all_policies(GPI_test_freq: int, num_GPI_iters : int, market_params: tuple, epsilon : float) -> list:
    # test the performance for the following GPI iterations
    iters_to_test = list(range(1, num_GPI_iters+1, GPI_test_freq))
    
    
    saved_stats = []
    for GPI_iter in iters_to_test:
        print(f"Testing the Performance after GPI iteration {GPI_iter}")
        

        q_table_string = CONFIG['setup'] + f'\\q_tables\\q_table_seller_after_GPI_{GPI_iter}.csv'
        market_params[3]['sellers'][1][2]['q_table_seller'] = q_table_string    
        market_params[3]['sellers'][1][2]['epsilon'] = 0.0
        print("using q_table: ", q_table_string)


        cumulative_stats = test_policy(episodes=CONFIG['test_episodes'], market_params=market_params)

        
        epsilon = epsilon_decay('linear', GPI_iter, CONFIG["num_GPI_iter"], CONFIG['epsilon'], CONFIG["epsilon_min"])
        

        for ttype in cumulative_stats:
            print(f"Performance Test: GPI Iter {GPI_iter}, {ttype} average profit: {cumulative_stats[ttype]['avg_profit']}")
            
        saved_stats.append(cumulative_stats)
    
    return saved_stats # saved stats is a list of dictionaries one for each GPI iteration.





