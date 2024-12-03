import argparse
import numpy as np
import random
from collections import deque

from environment import MountainCar, GridWorld

from typing import Union, Tuple, Optional # for type annotations

"""
Please read: THE ENVIRONMENT INTERFACE

In this homework, we provide the environment (either MountainCar or GridWorld) 
to you. The environment returns states, represented as 1D numpy arrays, rewards, 
and a Boolean flag indicating whether the episode has terminated. The environment 
accepts actions, represented as integers.

The only file you need to modify/read is this one. We describe the environment 
interface below.

class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 
        make sure that this is turned off when you submit to the autograder.

        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.
    
    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.
            
            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.
    
    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

Then, you can initialize your weight matrix to all zeroes with shape 
(env.action_space, env.state_space+1) (if you choose to fold the bias term in). 
Note that the states returned by the environment do *NOT* have the bias term 
folded in.
"""

def set_seed(seed: int):
    '''
    DO NOT MODIFY THIS FUNCITON.
    Sets the numpy random seed.
    '''
    np.random.seed(seed)
    random.seed(seed)


def round_output(places: int):
    '''
    DO NOT MODIFY THIS FUNCTION.
    Decorator to round output of a function to certain 
    number of decimal places. You do not need to know how this works.
    '''
    def wrapper(fn):
        def wrapped_fn(*args, **kwargs):
            return np.round(fn(*args, **kwargs), places)
        return wrapped_fn
    return wrapper


def parse_args() -> Tuple[str, str, str, str, int, int, float, float, float, int, int, int]:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
        (10) replay_enabled : An integer, 0 = perform Q_learning without experience replay, 
                            1 = perform Q_learning with experience replay
        (11) buffer_size : An integer representing the max size of the replay buffer
        (12) batch_size : An integer representing the size of the bath that should be sampled
                          from the replay buffer
    
    Usage:
        (env_type, mode, weight_out, returns_out, 
         episodes, max_iterations, epsilon, gamma, lr,
         replay_enabled, buffer_size, batch_size) = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("replay_enabled", type=int, choices=[0, 1])
    parser.add_argument("buffer_size", type=int)
    parser.add_argument("batch_size", type=int)

    args = parser.parse_args()

    return (args.env, args.mode, args.weight_out, args.returns_out, 
            args.episodes, args.max_iterations, 
            args.epsilon, args.gamma, args.learning_rate,
            args.replay_enabled, args.buffer_size, args.batch_size)

class ExperienceReplay:
    def __init__(self, buffer_size):
        '''
        Initialize the replay buffer
        Parameters:
            buffer_size (int) : Maximum size of the replay buffer
        '''
        # TODO: initialize the replay buffer with the correct size (Hint: use a deque)
        self.replay_buffer = deque(maxlen=buffer_size)
        
    
    def add(self, state, action, reward, next_state):
        '''
        Add the experience tuple to the replay buffer
        Parameters:
            state       (np.ndarray): State encoded as vector with shape (state_space,).
            action             (int): Action taken. Satisfies 0 <= action < action_space.
            reward             (int): Reward received from taking the action from the current state
            next_state  (np.ndarray): Next state encoded as vector with shape (state_space,).
        '''
        # TODO: append the experience to the replay buffer
        experience = (state, action, reward, next_state)
        self.replay_buffer.append(experience)
    
    def sample(self, batch_size):
        '''
        Return a randomly sampled batch of experiences from the replay buffer,
        only if there are enough experiences in the buffer
        Parameters:
            batch_size  (int): Number of randomly sampled experiences to return
        Returns:
            Return array of experiences
        '''

        # TODO: return a randomly sampled batch of experiences, only if there
        #       are enough experiences in the buffer
        if len(self.replay_buffer) >= batch_size:
            return random.sample(self.replay_buffer, batch_size)
        else:
            return None
            #raise ValueError(f"Not enough experience in replay buffer to sample {batch_size} samples.")

    

@round_output(5) # DON'T DELETE THIS LINE
def Q(W: np.ndarray, state: np.ndarray, 
      action: Optional[int] = None) -> Union[float, np.ndarray]:
    '''
    Helper function to compute Q-values for function-approximation 
    Q-learning.

    Note: Do not delete or move the `@round_output(5)` line on top of 
          this function. This just ensures that your Q-value is rounded to 5
          decimal places, which avoids some *pernicious* cross-platform 
          rounding errors.

    Parameters:
        W     (np.ndarray): Weight matrix with folded-in bias with 
                            shape (action_space, state_space+1).
        state (np.ndarray): State encoded as vector with shape (state_space,).
        action       (int): Action taken. Satisfies 0 <= action < action_space.

    Returns:
        If action argument was provided, returns float Q(state, action).
        Otherwise, returns array of Q-values for all actions from state,
        [Q(state, a_0), Q(state, a_1), Q(state, a_2)...] for all a_i.
    '''
    # TODO: Implement this!
    state_with_one = np.insert(state, 0, 1)
    q_value = W @ state_with_one
    if action is not None:
        return q_value[action]
    else:
        return q_value


def selMaxAction(action_sel_arr: []):
    assert len(action_sel_arr) > 0

    if len(action_sel_arr) == 1:
        return 0

    action = 0
    q_value_max = action_sel_arr[0]
    for index in range(1, len(action_sel_arr)):
        if action_sel_arr[index] > q_value_max:
            action = index
            q_value_max = action_sel_arr[index]
    return action

if __name__ == "__main__":
    set_seed(10301) # DON'T DELETE THIS

    # Read in arguments
    (env_type, mode, weight_out, returns_out, 
     episodes, max_iterations, epsilon, gamma, lr,
     replay_enabled, buffer_size, batch_size) = parse_args()

    # Create environment
    if env_type == "mc":
        env = MountainCar(mode=mode, debug=False)# TODO: Replace me!
    elif env_type == "gw":
        env = GridWorld(mode="tile", debug=False)# TODO: Replace me!
    else: 
        raise Exception(f"Invalid environment type {env_type}")

    # TODO: Initialize your weight matrix. Remember to fold in a bias!
    W = np.zeros((env.action_space, env.state_space+1))

    if replay_enabled:
        experience_replay = ExperienceReplay(buffer_size)

    total_reward_episode = []

    for episode in range(episodes):

        # TODO: Get the initial state by calling env.reset()
        state_episode = env.reset()
        total_reward = 0

        for iteration in range(max_iterations):

            # TODO: Select an action based on the state via the epsilon-greedy 
            #       strategy.
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice([i for i in range(env.action_space)])
            else:
                action = selMaxAction(Q(W, state_episode))

            # TODO: Take a step in the environment with this action, and get the 
            #       returned next state, reward, and done flag.
            nxt_st, reward, done = env.step(action)
            total_reward += reward

            # TODO: If experience replay is disabled, use the original state,the
            #       action, the next state, and the reward to update the 
            #       parameters. Don't forget to update the bias term!
            #       If experience replay is enabled, sample experiences from the
            #       replay buffer to perform the update instead!
            if (replay_enabled == 1):
                experience_replay.add(state_episode, action, reward, nxt_st)
                sample_batches_exp = experience_replay.sample(batch_size)
                if sample_batches_exp is not None:
                    for s_, a_, r_, s_next_ in sample_batches_exp:
                        temperal_diff = Q(W, s_, a_) - (r_ + gamma*np.max(Q(W, s_next_)))
                        grad_q = np.insert(s_, 0, 1)
                        W[a_] -= lr*temperal_diff*grad_q
            else:
                #if done:
                    #temperal_diff = Q(W, state_episode, action) - reward
                #else:
                temperal_diff = Q(W, state_episode, action) - (reward + gamma*np.max(Q(W, nxt_st)))
                grad_q = np.insert(state_episode, 0, 1)
                W[action] -= lr*temperal_diff*grad_q

            # TODO: Remember to break out of this inner loop if the environment 
            #       signals done!
            if done:
                break

            state_episode = nxt_st
        total_reward_episode.append(total_reward)
            
    # TODO: Save your weights and returns. The reference solution uses 
    np.savetxt(weight_out, W, fmt="%.18e", delimiter=" ")
    np.savetxt(returns_out, total_reward_episode, fmt="%.18e", delimiter=" ")