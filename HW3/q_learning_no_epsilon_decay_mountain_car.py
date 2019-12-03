import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding

# Resources: 
# https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f
# https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e



def epsilon_greedy(Q, state, action_space, epsilon):
    # if in epsilon range use it 
    if np.random.rand() < 1 - epsilon:
        action = np.argmax(Q[state[0], state[1]])
    # else take random action 
    else:
        action = np.random.randint(0, action_space)
    return action 

def q_learning(learning_rate, discount, epsilon, episodes):
    # initialize environment
    env = gym.make("MountainCar-v0")
    env.reset()

    states = (env.observation_space.high - env.observation_space.low)*np.array([10,100])
    states = np.round(states, 0).astype(int) + 1

    # Q(s,a)
    Q_table = np.random.uniform(low = -1, high = 1, size = (states[0], states[1], env.action_space.n))


    reward_list = [] 
    var_list = [] 
    avg_reward_list = [] 

    # Q learning main loop 
    for i in range(episodes):
        finished = False
        total_reward = 0 
        reward = 0 
        state = env.reset()

        state_adj = (state - env.observation_space.low)*np.array([10,100])
        state_adj = np.round(state_adj, 0).astype(int)

        while not finished:
            # render last N episodes
            # comment out to see plots 
            # if i >= episodes - 1:
            #     env.render()

            # action = epsilon_greedy(Q_table, state_adj, env.action_space.n, epsilon)
            # pick aciton greedily without randomness
            action = np.argmax(Q_table[state_adj[0], state_adj[1]])

            next_state, reward, finished, info = env.step(action)

            # Discretize
            next_state_adj = (next_state - env.observation_space.low)*np.array([10,100])
            next_state_adj = np.round(next_state_adj, 0).astype(int)

            if finished and next_state[0] >= 0.5: # and ... condition 
                Q_table[state_adj[0], state_adj[1], action] = reward
            else:
                update = learning_rate * (reward + discount * np.max(Q_table[next_state_adj[0],next_state_adj[1]])
                        - Q_table[state_adj[0], state_adj[1], action])
                # update Q table
                Q_table[state_adj[0], state_adj[1], action] += update

            
            total_reward += reward
            state_adj = next_state_adj

        reward_list.append(total_reward)

        if (i) % 100 == 0:
            avg_reward = np.mean(reward_list)
            var = np.var(reward_list)
            var_list.append(var)
            avg_reward_list.append(avg_reward)
            reward_list = []
                
    env.close()
    return (avg_reward_list, var_list)

# Adjust these parameters as needed 
number_of_episodes = 2500
learning_rate = 0.1
gamma = 0.9
epsilon = 0.8

def single_run():
    """
    Run the algorithm once 
    """
    rewards_and_var = q_learning(learning_rate, gamma, epsilon, number_of_episodes)
    avg_reward = rewards_and_var[0]
    var = rewards_and_var[1]
    episodes1 = 100*(np.arange(len(avg_reward)) + 1)
    episodes2 = 100*(np.arange(len(var)) + 1)
    plt.figure("Average Reward vs. Episodes")
    plt.title("Average Reward vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.plot(episodes1, avg_reward, color='blue')
    plt.figure("Variance vs. Episodes")
    plt.title("Variance vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Variance")
    plt.plot(episodes2, var, color='orange')
    plt.figure("Average Reward w/ Variance vs. Episodes")
    plt.title("Average Reward w/ Variance vs. Episodes") 
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward w/ Variance")
    plt.errorbar(episodes1, avg_reward, var, linestyle='None', marker='^', ecolor="orange")
    plt.show()


def multi_run(N):
    """
    Run the algorithm N times 
    @param N - number of times to test (e.g. 20)
    """
    rewards = [] 
    vars = [] 
    for _ in range(N):
        rewards_and_var = q_learning(learning_rate, gamma, epsilon, number_of_episodes)
        avg_reward = rewards_and_var[0]
        var = rewards_and_var[1]
        rewards.append(avg_reward)
        vars.append(var)

    rewards = list(zip(*rewards))
    vars = list(zip(*vars))
    
    reward_to_plot = []
    for sublist in rewards:
        reward_to_plot.append(np.mean(sublist))
    var_to_plot = [] 
    for sublist in vars:
        var_to_plot.append(np.mean(sublist))
    
    episodes1 = 100*(np.arange(len(avg_reward)) + 1)
    episodes2 = 100*(np.arange(len(var)) + 1)
    plt.figure("Average Reward vs. Episodes")
    plt.title("Average Reward vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.plot(episodes1, reward_to_plot, color='blue')
    plt.savefig("q_learning_no_epsilon_results/Average_Reward_vs_Episodes.png")
    plt.figure("Variance vs. Episodes")
    plt.title("Variance vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Variance")
    plt.plot(episodes2, var_to_plot, color='orange')
    plt.savefig("q_learning_no_epsilon_results/Variance_vs_Episodes.png")
    plt.figure("Average Reward w/ Variance vs. Episodes")
    plt.title("Average Reward w/ Variance vs. Episodes") 
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward w/ Variance")
    plt.errorbar(episodes1, reward_to_plot, var_to_plot, linestyle='None', marker='^', ecolor="orange")
    plt.savefig("q_learning_no_epsilon_results/Average_Reward_and_Variance_vs_Episodes.png")


# choose multi or single run
# single_run()
multi_run(20)