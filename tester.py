# Original state
state = ('(3.0, 3.0, 3.0)', 1.0)

# Extract the string representing the tuple and the action
state_str, action = state

# Convert the string to a list of strings (remove parentheses and split by ', ')
string_tuple = tuple(state_str.strip('()').split(', '))

# Inject 'Seller' at the beginning
new_tuple = ('Seller',) + string_tuple

# Combine with the original action
new_state = (new_tuple, action)

print(new_state)