import matplotlib.pyplot as plt
from collections import defaultdict
import os

from config.config_params import CONFIG

# A file for plotting

def plot_avg_profit(saved_stats):
    profits = defaultdict(list) # initialise empty dictionary to save profits for each trader
    
    # iterate through the list of saved_stats, each corresponding to a GPI iteration
    for cumulative_stats in saved_stats:
        for ttype in cumulative_stats:
            profits[ttype].append(cumulative_stats[ttype]['avg_profit'])
            

    for key, value in profits.items(): #key is trader type and value is the list of profits
        GPI_idexes = range(1, len(value)+1)
        plt.plot(GPI_idexes, value, label=key)
        
    plt.grid(False)
    plt.xlabel('GPI Iteration')
    plt.ylabel('Average Profit')
    plt.title('Average Profit per Trader Type Over GPI Iterations')
    plt.legend(title='Trader Type')
    
    # Save the plot to the Plots_Folder directory
    plot_filename = os.path.join(CONFIG["plots"], 'Profits.png')
    plt.savefig(plot_filename)
    
    # Show the plot
    plt.show()
    
    
def create_plots(saved_stats):
    plot_avg_profit(saved_stats)