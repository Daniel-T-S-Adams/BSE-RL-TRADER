def epsilon_decay(strat, timestep, max_timestep, eps_start=1.0, eps_min=0.05, eps_decay=0.99):

    if strat == 'constant':
        epsilon = 0.9

    if strat == 'linear':
        decay_steps = eps_decay * max_timestep
        epsilon = max(eps_min, eps_start - (eps_start - eps_min) * min(1.0, timestep / decay_steps))

    if strat == 'exponential':
        epsilon = max(eps_min, eps_decay*epsilon)

    return epsilon


# strat = 'linear'
# max_timestep = 100
# eps_decay = 0.8

