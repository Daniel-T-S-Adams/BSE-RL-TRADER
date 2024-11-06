

def linear_epsilon_decay(timestep, max_timestep, eps_start=1.0, eps_min=0.05, eps_decay=0.9):
    
    """
    Linearly decays epsilon from a starting value to a minimum value over a specified portion of the 
    total timesteps, controlled by a decay factor.

    Parameters:
        timestep (int): The current timestep in training.
        max_timestep (int): The total number of timesteps over the training duration.
        eps_start (float): The initial epsilon value at the beginning of training (default is 1.0).
        eps_min (float): The minimum epsilon value the decay can reach (default is 0.05).
        eps_decay (float): The fraction of max_timestep over which epsilon will linearly decay to eps_min 
                           (default is 0.9). A value closer to 1 means slower decay, while lower values mean 
                           faster decay.

    Returns:
        float: The decayed epsilon value for the current timestep.

    Functionality:
        - Calculates the number of steps over which epsilon should decay based on eps_decay and max_timestep.
        - Linearly reduces epsilon from eps_start to eps_min over the calculated decay steps.
        - If timestep exceeds decay steps, epsilon remains at eps_min.

    Example:
        With eps_decay = 0.9, epsilon will reach eps_min at 90% of max_timestep and stay constant afterward.
    """
    
    
    decay_steps = eps_decay * max_timestep
    epsilon = max(eps_min, eps_start - (eps_start - eps_min) * min(1.0, timestep / decay_steps))

    return epsilon


# strat = 'linear'
# max_timestep = 100
# eps_decay = 0.8


