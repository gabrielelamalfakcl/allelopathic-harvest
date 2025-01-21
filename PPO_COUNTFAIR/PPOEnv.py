import random
import numpy as np
from AgentsPolicy import RandomMovementPolicy, StationaryPolicy
from BerryRegrowth import LinearRegrowth, CubicRegrowth
from collections import deque

class Player:
    class Preference:
        def __init__(self, berry_type):
            self.berry_type = berry_type

        def get_preference(self):
            return self.berry_type

    def __init__(self, name, x, y, policy, preference_berry_type, has_sensitive=False, verbose=False):
        self.name = name
        self.x = x
        self.y = y
        self.reward = 0
        self.policy = policy
        self.preference = self.Preference(preference_berry_type)
        self.has_sensitive = has_sensitive
        self.policy_counter = 0
        self.is_obstacolated = False
        self.obstacolation_cooldown = 0
        self.state_history = []
        self.move_counter = 0
        self.last_action_position = None
        self.verbose = verbose  # Add verbose flag

    def move(self, environment, direction):
        """
        Moves the agent in the specified direction within the environment.
        - If the agent is obstructed, the move is ignored, the flag is reset, and 'stay' is returned.
        - If the agent is not obstructed and the move is within the boundaries of the environment,
        the agent's position is updated accordingly.
        - If the agent's new position is empty, the agent is moved to the new position and the previous
        position is set to None.
        """
        # Define all possible directions and their corresponding coordinate changes
        possible_moves = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0)
        }
        
        # Check if direction is valid
        if direction not in possible_moves:
            return False
        
        # Calculate new position based on the chosen direction
        dx, dy = possible_moves[direction]
        new_x, new_y = self.x + dx, self.y + dy
        
        # Check if the new position is within the grid and there are no obstacles
        if 0 <= new_x < environment.x_dim and 0 <= new_y < environment.y_dim:
            if environment.grid[new_x][new_y] is None: # Check if the new position is empty
                environment.grid[self.x][self.y] = None
                self.x, self.y = new_x, new_y
                environment.grid[self.x][self.y] = self # Update the grid with the new position
                self.reward = 0
                return True
        else:
            return False
        
    def ripe_eat_fruit(self, environment):
        """
        The agent check whether there are bushes around and targets only the first bush it finds in its adjacent cells.
        2 actions at disposal: ripe and eat. If a bush is not ripe then the berry cannot be eaten.
        If a berry is eaten, the berry will be reset.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # Check if there are bushes in the adjacent cells
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                target_entity = environment.grid[target_x][target_y]      
                # If a bush is found, interact with it
                if isinstance(target_entity, Bush):
                    if target_entity.is_ripe:
                        self.eat_fruit(target_entity)
                        break
                    else:
                        self.reward = self.ripe(target_entity)
                        
        return self.reward
                      
    def change_bush_color(self, environment):
        """
        Change the color of a bush in the adjacent cells if bush is not of color preference.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                target_entity = environment.grid[target_x][target_y]
                if isinstance(target_entity, Bush) and target_entity.berry_type != self.preference.get_preference():
                    self.reward = self.change_color(target_entity)
                    break # Stop after interacting with the first bush
        
        return self.reward
                    
    def interact_with_nearby_player(self, environment):
        """
        Interact with a player in the adjacent cells and obstruct them.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            
            # Check if the target position is within the grid bounds
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                target_entity = environment.grid[target_x][target_y]
                
                # Check if the entity is a player
                if isinstance(target_entity, Player):
                    self.reward = self.obstacolate_player(target_entity)
                    break  # Stop after interacting with the first player
                
        return self.reward
                
    def eat_fruit(self, bush):
        """
        Eat a berry from a bush.
        """
        if bush.berry_type is not None: # Check if the bush has a berry
            if bush.berry_type == self.preference.get_preference():
                self.reward = 5
            else:
                self.reward = 1
        
        # Reset the berry once it is eaten
        bush.current_berry_type = None # Reset the generic berry type
        bush.is_ripe = False
        bush.time_step = 0
        
        return self.reward
                
    def ripe(self, bush):
        """
        Ripen a berry on a bush.
        """
        if bush.is_ripe:
            self.reward = 0
        else:
            if bush.berry_type == self.preference.get_preference():
                self.reward = 3
            else:
                self.reward = 1
        # Bush becomes ripe
        bush.is_ripe = True
        
        return self.reward
        
    def change_color(self, bush, new_color=None):
        """
        Changes the color of the berry currently on the bush and also changes 
        the bush's future berry production to this new color.
        """
        old_color = bush.berry_type
        if new_color is None:
            # If no new color is provided, use the agent's preference
            new_color = self.preference.get_preference()

        # Change the current berry color
        bush.berry_type = new_color
                
        # Reset the bush's growth cycle
        bush.is_ripe = False
        bush.time_step = 0
        
        # Reward the agent for changing the color
        if old_color != self.preference.get_preference():
            self.reward = 5
        else:
            self.reward = 0
        
        return self.reward
        
    def obstacolate_player(self, other_player):
        """
        Obstacolate a player.
        """
        other_player.is_obstacolated = True
        self.obstacolation_cooldown = 1
        other_player.obstacolation_cooldown = 1
        if other_player.preference.get_preference() != self.preference.get_preference():
            self.reward = 3  # Increased reward for obstacolating a different-preference player
        else:
            self.reward = 0  # Slightly reduced penalty for same-preference obstacolation
        
        return self.reward
        
    def get_nearest_bush_distance(self, environment, bush_type):
        distances = [
            abs(self.x - bush.x) + abs(self.y - bush.y)
            for bush in environment.bushes if bush.berry_type == bush_type
        ]

        return min(distances) if distances else environment.x_dim + environment.y_dim
    
    def get_nearest_player_distance(self, environment):
        distances = [
            abs(self.x - player.x) + abs(self.y - player.y)
            for player in environment.players if player != self
        ]

        return min(distances) if distances else environment.x_dim + environment.y_dim

    def get_state(self, environment):
        """
        Returns the state of the player as a numpy array.
        (state = [x, y, has_sensitive, reward, preference])
        We also add the following global information:
        a) count of red and blue bushes
        b) nearest bush distance
        c) nearest player distance
        d) history
        """
        state = [self.x, self.y, self.has_sensitive, self.reward]
        preference_num = 1 if self.preference.get_preference() == "red" else 2 # 1 for red, 2 for blue
        state.append(preference_num)
        
        # Bush counting
        red_bush_count = 0
        blue_bush_count = 0
        for bush in environment.bushes:
            if bush.berry_type == 'red':
                red_bush_count += 1
            elif bush.berry_type == 'blue':
                blue_bush_count += 1
        state.extend([red_bush_count, blue_bush_count])

        # Nearest bush distance
        red_bush_distance = self.get_nearest_bush_distance(environment, 'red')
        blue_bush_distance = self.get_nearest_bush_distance(environment, 'blue')
        state.extend([red_bush_distance, blue_bush_distance])

        # Nearest player distance
        nearest_player_distance = self.get_nearest_player_distance(environment)
        state.append(nearest_player_distance)

        # history
        # current_state = [self.x, self.y, self.has_sensitive, self.reward, preference_num, red_bush_count, blue_bush_count, red_bush_distance, blue_bush_distance, nearest_player_distance]
        # # Maintain a fixed-length history (3 states)
        # if len(self.state_history) >= 3:
        #     self.state_history.pop(0)
        # self.state_history.append(current_state)

        # # If the history has less than 3 states, pad with zeros
        # while len(self.state_history) < 3:
        #     self.state_history.insert(0, [0] * len(current_state))

        # # Flatten the history list
        # history_flattened = [item for sublist in self.state_history for item in sublist]

        # # Combine current state with history
        # full_state = state + history_flattened
        
        return np.array(state, dtype=np.float32)

    # def update_rewards(self, move_reward):
    #     """
    #     Updates the rewards for the player.
    #     For now we do not use it.
    #     """      
        # normalized_reward = move_reward #self.normalise_rewards(move_reward)
        
        # # Apply a decay factor if the player has not moved for some time
        # if self.move_counter > 5:
        #     decay_factor = 0.99 ** (self.move_counter - 5)
        # else:
        #     decay_factor = 1
        
        # if self.has_sensitive:
        #     # Maybe some actions get less reward, but interaction-heavy tasks are not penalized
        #     if self.current_action in ["ripe_eat_fruit", "interact_with_nearby_player"]:
        #         action_frequency_scale = 1  # Full reward for important interactions
        #     else:
        #         action_frequency_scale = 0.7  # Slightly reduce reward for less frequent actions
        # else:
        #     action_frequency_scale = 1

        # # Final reward calculation with more controlled decay and sensitive attribute scaling
        # self.normalized_reward = normalized_reward * decay_factor * action_frequency_scale
        # self.reward += self.normalized_reward

    def can_interact_with_bush(self, environment):
        """
        Checks if there's a bush in the adjacent cells (up, down, left, right) that the player can interact with.
        Returns True if such a bush exists, otherwise returns False.
        """
        # Define the possible directions to check for a nearby bush
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Loop through each direction to see if a bush is adjacent
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            # Check if the target position is within the bounds of the grid
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                target_entity = environment.grid[target_x][target_y]
                # Check if the entity at the target position is a bush
                if isinstance(target_entity, Bush):
                    if self.verbose:
                        print("Bush found in adjacent cell at position ({}, {})".format(target_x, target_y))
                    return True
        
        # If no bushes are found in the adjacent cells, return False
        return False

    def can_interact_with_player(self, environment):
        """
        Checks if there's a player in the adjacent cells (up, down, left, right) that the player can interact with.
        Returns True if such a bush exists, otherwise returns False.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            target_x, target_y = self.x + dx, self.y + dy
            if 0 <= target_x < environment.x_dim and 0 <= target_y < environment.y_dim:
                target_entity = environment.grid[target_x][target_y]
                if isinstance(target_entity, Player):
                    if self.verbose:
                        print("Player found in adjacent cell at position ({}, {})".format(target_x, target_y))
                    return True
        
        # If no players are found in the adjacent cells, return False
        return False

    def execute_policy_with_action(self, environment, action: int):
        num2action = {0: "stay", 1: "move_up", 2: "move_down", 3: "move_left", 4: "move_right", 5: "ripe_eat_fruit", 6: "change_bush_color", 7: "interact_with_nearby_player"}
        available_actions = list(num2action.keys())  # Set of all possible actions
        self.current_action = num2action[action]  # Store the current action
        self.reward = 0  # Reset immediate reward for each action
        retry_limit = 3  # Limit the number of retries to 3
        retries = 0  # Initialize retry counter
        self.move_counter += 1

        # If the player is obstructed, no reward is given, just reduce the cooldown
        if self.is_obstacolated:
            self.obstacolation_cooldown -= 1
            return self.reward  # No reward

        # If the player has a sensitive attribute, they may skip some actions
        if self.has_sensitive and (self.move_counter % 2) != 0:
            return self.reward  # No reward

        # Check if the action is valid or not
        while retries < retry_limit:
            retries += 1
            if action == 0:  # Stay
                return self.reward  # No reward for staying
            
            # Movement actions
            if action in [1, 2, 3, 4]:  # Move actions
                success = self.move(environment, num2action[action])
                if success:  # Reward if the move was successful
                    return self.reward
                else:
                    available_actions.remove(action)  # Remove invalid action
                    action = random.choice(available_actions) if available_actions else None
                    if action is None:
                        return self.reward
            
            # Bush interactions (ripe/eat fruit)
            if action == 5:  # ripe_eat_fruit
                if self.can_interact_with_bush(environment):
                    self.reward = self.ripe_eat_fruit(environment)
                    return self.reward
                else:
                    available_actions.remove(action)  # Remove invalid action
                    action = random.choice(available_actions) if available_actions else None
                    if action is None:
                        return self.reward  # Return 0 if no valid action
            
            # Bush color change
            elif action == 6:  # change_bush_color
                if self.can_interact_with_bush(environment):
                    self.reward = self.change_bush_color(environment)
                    return self.reward
                else:
                    available_actions.remove(action)  # Remove invalid action
                    action = random.choice(available_actions) if available_actions else None
                    if action is None:
                        return self.reward  # Return 0 if no valid action
            
            # Player interaction (obstacolate)
            elif action == 7:  # interact_with_nearby_player
                if self.can_interact_with_player(environment):
                    self.reward = self.interact_with_nearby_player(environment)
                    return self.reward
                else:
                    available_actions.remove(action)  # Remove invalid action
                    action = random.choice(available_actions) if available_actions else None
                    if action is None:
                        return self.reward  # Return 0 if no valid action

        return self.reward

        
class Bush:
    def __init__(self, x, y, berry_type, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate):
        self.x = x
        self.y = y
        self.berry_type = berry_type # Generic bush category (e.g., if "red" then it will produce red berries)
        self.current_berry_type = None # Current berry type on the bush (e.g., "red" or "blue" berry)
        self.regrowth_rate = regrowth_rate # Time steps required for the bush to regrow berries
        self.time_step = 0
        self.is_ripe = False
        #regrowth and lifespan
        self.regrowth_function = regrowth_function # Regrowth function determines which berry type the bush will produce
        self.max_lifespan = max_lifespan # Maximum lifespan of the bush
        self.spont_growth_rate = spont_growth_rate # Spontaneous growth rate for new bush
        
        self.time_step = 0
        self.lifespan = 0

    def update(self):
        """
        Updates the bush state.
        """
        # Check if the bush has reached the lifespan
        self.lifespan += 1
        if self.lifespan >= self.max_lifespan:
            return False # Bush died
        
        # Regrowth logic: once the time step reaches the regrowth rate, the bush will regrow berries
        if self.current_berry_type is not None:
            self.time_step += 1
            if self.time_step >= self.regrowth_rate:
                self.current_berry_type = self.berry_type
                self.is_ripe = True
                self.time_step = 0
        
        return True # Bush is still alive


class Environment:
    def __init__(self, x_dim, y_dim, max_steps, num_players, num_bushes, 
                 red_player_percentage, blue_player_percentage, 
                 red_bush_percentage, blue_bush_percentage, 
                 sensitive_percentage, max_lifespan, regrowth_rate, spont_growth_rate, verbose=False, random_seed=None):
        # Set random seed if provided for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.max_steps = max_steps
        self.num_players = num_players
        self.num_bushes = num_bushes
        self.red_player_percentage = red_player_percentage
        self.blue_player_percentage = blue_player_percentage
        self.red_bush_percentage = red_bush_percentage
        self.blue_bush_percentage = blue_bush_percentage
        self.sensitive_percentage = sensitive_percentage
        self.grid = [[None for _ in range(y_dim)] for _ in range(x_dim)]
        self.players = []
        self.bushes = []
        self.current_step = 0
        self.last_action_position = None
        self.max_lifespan = max_lifespan
        self.regrowth_rate = regrowth_rate
        self.spont_growth_rate = spont_growth_rate
        self.verbose = verbose

    def add_player(self, name, x, y, policy, preference_berry_type, has_sensitive=False):
        """
        Adds a player to the environment.
        """
        player = Player(name, x, y, policy, preference_berry_type, has_sensitive)
        self.players.append(player)
        self.grid[x][y] = player

    def add_bush(self, x, y, berry_type, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate):
        """
        Adds a bush to the environment.
        """
        bush = Bush(x, y, berry_type, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)
        self.bushes.append(bush)
        self.grid[x][y] = bush

    def update_bushes(self):
        """
        Bushes update their state in the environment.
        """
        for bush in self.bushes[:]:  # Iterate over the bushes list
            if not bush.update():  # If the bush has reached the end of its lifespan
                self.bush_lifespan_end(bush)  # Remove the bush

    def bush_spontaneous_growth(self, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate):
        """
        Bushes grow spontaneously in the environment.
        The color of the bush is determined by the number of red and blue bushes present.
        If no red or blue bushes are present, the new bush will have no color.
        """
        # Count the number of red and blue bushes
        red_bush_count = sum(1 for bush in self.bushes if bush.berry_type == 'red')
        blue_bush_count = sum(1 for bush in self.bushes if bush.berry_type == 'blue')
        
        # Determine the color of the new bush based on the existing bushes
        if red_bush_count == 0 and blue_bush_count == 0:
            berry_type = None
        else:
            berry_type = regrowth_function(red_bush_count, blue_bush_count)
        
        # Place the bush in an available random position if possible
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(0, self.x_dim - 1)
            y = random.randint(0, self.y_dim - 1)
            if self.grid[x][y] is None:
                self.add_bush(x, y, berry_type, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)
                break       

    def bush_lifespan_end(self, bush):
        """
        Bushes die spontaneously in the environment after their lifespan ends.
        """
        self.grid[bush.x][bush.y] = None
        self.bushes.remove(bush)

    def randomize_positions(self, regrowth_rate, max_lifespan, spont_growth_rate):
        """
        Randomizes the positions of players and bushes in the environment.
        """
        # Get all available positions in the grid (this function is only used when reset is called)
        available_positions = [(x, y) for x in range(self.x_dim) for y in range(self.y_dim)]
        random.shuffle(available_positions)
        
        # Initialize players
        num_red_players = int(self.num_players * self.red_player_percentage)
        num_blue_players = int(self.num_players * self.blue_player_percentage)
        num_players_with_sensitive = int(self.num_players * self.sensitive_percentage)
        
        player_indices = list(range(self.num_players))
        random.shuffle(player_indices)
        players_with_sensitive = set(player_indices[:num_players_with_sensitive])
        
        # Add blue and red players if positions are available
        for i in range(num_red_players):
            if not available_positions:
                print(f"Failed to place red player {i+1}: No available positions.")
                continue
            x, y = available_positions.pop()  # Choose a position randomly from available ones
            has_sensitive = i in players_with_sensitive
            policy = RandomMovementPolicy() if not has_sensitive else StationaryPolicy()
            self.add_player(f"Player{i+1}", x, y, policy, "red", has_sensitive)

        for i in range(num_blue_players):
            if not available_positions:
                print(f"Failed to place blue player {i+num_red_players+1}: No available positions.")
                continue
            x, y = available_positions.pop()
            has_sensitive = (i + num_red_players) in players_with_sensitive
            policy = RandomMovementPolicy() if not has_sensitive else StationaryPolicy()
            self.add_player(f"Player{i+num_red_players+1}", x, y, policy, "blue", has_sensitive)
        
        # Initialise bushes
        num_red_bushes = int(self.num_bushes * self.red_bush_percentage)
        num_blue_bushes = int(self.num_bushes * self.blue_bush_percentage)
        
        # Add red and blue bushes if positions are available
        for i in range(num_red_bushes):
            if not available_positions:
                print(f"Failed to place red bush {i+1}: No available positions.")
                continue
            x, y = available_positions.pop()
            regrowth_function = LinearRegrowth().regrowth
            self.add_bush(x, y, "red", regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)
            
        for i in range(num_blue_bushes):
            if not available_positions:
                print(f"Failed to place blue bush {i+1}: No available positions.")
                continue
            x, y = available_positions.pop()
            regrowth_function = LinearRegrowth().regrowth
            self.add_bush(x, y, "blue", regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)      
        
    def update_players(self):
        #print(f"Updating players at step {self.current_step}...")
        for player in self.players:
            if player.obstacolation_cooldown > 0:
                player.obstacolation_cooldown -= 1
            #print(f"Player {player.name} is at ({player.x}, {player.y}) with total reward {player.reward}")

    def reset(self):
        """
        Resets the environment. This function is used at the beginning of each epoch.
        """
        self.grid = [[None for _ in range(self.y_dim)] for _ in range(self.x_dim)]
        self.players = []
        self.bushes = []
        self.current_step = 0
        self.randomize_positions(self.regrowth_rate, self.max_lifespan, self.spont_growth_rate)  # Use the stored parameters
        
        if self.verbose:
            self.print_matrix()
        
        return self.get_state(self)

    def get_state(self, environment):
        """
        Returns the state of the environment as a matrix where each player's state is a row.
        """
        states = [player.get_state(environment) for player in self.players]
        return np.vstack(states)

    def step(self, actions, environment, regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate):
        """
        Executes the actions for all players, updates the environment, and returns the next state, rewards, and whether the episode is done.
        """
        num_players = len(self.players)
        rewards = np.zeros(num_players, dtype=np.float32)

        # Iterate over the players and their corresponding actions
        for i, (player, action) in enumerate(zip(self.players, actions)):
            # Execute the action for the player and update the reward
            rewardfromaction = player.execute_policy_with_action(self, action)
            rewards[i] = rewardfromaction
            self.last_action_position = player.last_action_position

        # Update the environment: bushes, players, and growth cycles
        self.update_bushes()
        self.update_players()

        # Handle spontaneous bush growth if applicable (one bush grows every spont_growth_rate steps)
        if self.current_step % spont_growth_rate == 0:
            self.bush_spontaneous_growth(regrowth_rate, regrowth_function, max_lifespan, spont_growth_rate)

        # Increment the step counter
        self.current_step += 1

        # Get the next state of the environment
        next_state = self.get_state(environment)

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        if done:
            print(f"Episode finished after {self.current_step} steps")
        
        if self.verbose == True:
            self.print_matrix()
            print(f"Step {self.current_step} completed. Rewards: {rewards}")

        return next_state, rewards, done

    # rendering methods
    def render_matrix(self):
        # Create a matrix filled with '.' to represent empty cells
        matrix = [['.' for _ in range(self.x_dim)] for _ in range(self.y_dim)]

        # Place bushes in the matrix with detailed info
        for bush in self.bushes:
            symbol = ''
            if bush.berry_type == 'red':
                if bush.is_ripe:
                    if bush.current_berry_type is not None:
                        symbol = 'R(R)(F)'  # Red bush ripened with Fruit
                    else:
                        symbol = 'R(R)(NF)'  # Red bush ripened without Fruit
                else:
                    if bush.current_berry_type is not None:
                        symbol = 'R(U)(F)'  # Red bush unripened with Fruit
                    else:
                        symbol = 'R(U)(NF)'  # Red bush unripened without Fruit
            elif bush.berry_type == 'blue':
                if bush.is_ripe:
                    if bush.current_berry_type is not None:
                        symbol = 'B(R)(F)'  # Blue bush ripened with Fruit
                    else:
                        symbol = 'B(R)(NF)'  # Blue bush ripened without Fruit
                else:
                    if bush.current_berry_type is not None:
                        symbol = 'B(U)(F)'  # Blue bush unripened with Fruit
                    else:
                        symbol = 'B(U)(NF)'  # Blue bush unripened without Fruit

            # Place the symbol in the matrix
            matrix[bush.y][bush.x] = symbol

        # Place players in the matrix
        for player in self.players:
            if player.preference.get_preference() == 'red':
                matrix[player.y][player.x] = 'r'  # Red player
            elif player.preference.get_preference() == 'blue':
                matrix[player.y][player.x] = 'b'  # Blue player

        return matrix

    def print_matrix(self):
        matrix = self.render_matrix()
        print("Environment:")

        cell_width = max(len(cell) for row in matrix for cell in row) + 2  # Calculate max cell width with padding

        for row in matrix:
            colored_row = []
            for cell in row:
                padded_cell = cell.center(cell_width)  # Center the text within the padded cell

                if cell == '.':
                    colored_cell = padded_cell  # Default color for empty cells
                elif cell.startswith('r'):  # Red player symbol
                    colored_cell = f"\033[91m{padded_cell}\033[0m"  # Red color
                elif cell.startswith('b'):  # Blue player symbol
                    colored_cell = f"\033[94m{padded_cell}\033[0m"  # Blue color
                elif cell.startswith('R'):  # Red berry bush
                    colored_cell = f"\033[91m{padded_cell}\033[0m"  # Red color
                elif cell.startswith('B'):  # Blue berry bush
                    colored_cell = f"\033[94m{padded_cell}\033[0m"  # Blue color
                else:
                    colored_cell = padded_cell  # Default color for other cells

                colored_row.append(colored_cell)

            print(' '.join(colored_row))

        print('Matrix printed')