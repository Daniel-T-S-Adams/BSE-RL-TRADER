#
# A Configuration file for the Global Variables.
#

# Imports from Standard Library 
import os
from collections import defaultdict

CONFIG = {
    # setup ID for this configuration
    'setup': 'Parameter_setup_1',
    
    # Parameters for the Agent 
    "eps_per_evaluation": 1,
    "num_GPI_iter": 5000,
    "GPI_CSV_save_freq": 1000,
    "test_episodes": 1000,
    "gamma": 0.3,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9,
    "alpha" : 0.1,
    "no._of_bins": 5,
    "initial q_table": defaultdict(lambda: 0.0),
    
    # Parameters for the BSE
    "bse_sys_minprice" : 1,
    "bse_sys_maxprice" : 5,
    
}
    
#
# Dictionary is initialised now can self reference if needed. 
#

#### Parameters for the Agent
CONFIG["total_eps"] = CONFIG["eps_per_evaluation"] * CONFIG["num_GPI_iter"] # total number of episodes
CONFIG["GPI_test_freq"] = CONFIG["GPI_CSV_save_freq"] * 1 # how often we test, must be a multiple of GPI_CSV_save_freq since we need to load CSV file for testing.
CONFIG["action_space"] =  [n/1.0 for n in range(CONFIG["bse_sys_maxprice"] - CONFIG["bse_sys_minprice"] + 1)]  # This allows any price on the BSE. Determined by the BSE min and max.

#### Parameters for the BSE

### Parameters for the Market

sess_id = 'session_1'
start_time = 0.0
end_time = 30.0

sellers_spec = [('GVWY',19), ('RL', 1, {'epsilon': CONFIG['epsilon_start'], 'action_space': CONFIG['action_space'], 'q_table_seller': CONFIG['initial q_table']})]
buyers_spec = [('GVWY',20)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

range1 = (1, 5)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (1, 5)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 30

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-fixed'}

dump_flags = {'dump_strats': True, 'dump_lobs': True, 'dump_avgbals': True, 'dump_tape': True, 'dump_blotters': True}
verbose = False

# save these to our config dictionary
CONFIG["market_params"] = (sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose)


### file paths
CONFIG['q_tables'] =  os.path.join(CONFIG["setup"], 'q_tables')
CONFIG['counts'] =  os.path.join(CONFIG["setup"], 'counts')
CONFIG['plots'] =  os.path.join(CONFIG["setup"], 'plots')
# Full path for the Parameters.py file (inside the setup_1 folder)
CONFIG['parameters'] = os.path.join(CONFIG["setup"], 'Parameters.py')

# Currently not in use 
### file paths
# folder path:
CONFIG['foler_path'] = f'BUYERS:_SELLERS:_'
# file path 
CONFIG['file_path'] = f'gamma-{CONFIG['gamma']}_GPIs-{CONFIG['num_GPI_iter']}_evals-{CONFIG['eps_per_evaluation']}' 
