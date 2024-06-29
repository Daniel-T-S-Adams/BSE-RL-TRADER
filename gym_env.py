import gymnasium as gym
from environment import AuctionEnv
from gymnasium import spaces

# Register the gymnasium environment
gym.register(
    id='Auction-v0',
    entry_point='__main__:AuctionEnv',
)


if __name__ == "__main__":
    # Create the LOB data structure with required format
    lob = {
        'bids': {'best': 28, 'worst': 22, 'lob': [(22, 1), (27, 2), (28, 1)]},
        'asks': {'best': 30, 'worst': 35, 'lob': [(30, 1), (32, 2), (35, 1)]}
    }
    
    # Create the orders list with required format
    orders = [(26,1)]  # Example order
    
    # Initialize the environment
    env = gym.make('Auction-v0', action_space=spaces.Discrete(100), obs_space=spaces.MultiDiscrete([24, 100, 5, 5, 5, 5]), gamma=1.0, epsilon=0.1, bins=10)
    
    # Set additional parameters
    env.set_additional_params(lob)
    env.orders = orders
    env.blotter = [{'time': 0, 'price': 25}, {'time': 1, 'price': 29}]  # Example blotter
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()  # Example random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

        if done:
            obs, _ = env.reset()  # Reset the environment when done


