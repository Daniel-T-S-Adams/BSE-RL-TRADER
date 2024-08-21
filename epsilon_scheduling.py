def epsilon_decay(strat, timestep, max_timestep, eps_start=1.0, eps_min=0.05):

    if strat == 'constant':
        epsilon = 0.9

    if strat == 'linear':
        epsilon_step = (eps_start - eps_min)/max_timestep
        epsilon = eps_start - timestep*epsilon_step
        
    return epsilon


# strat = 'linear'
# max_timestep = 100
# eps_decay = 0.8


