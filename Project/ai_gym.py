import gym
import numpy as np
import random
from IPython.display import clear_output
import itertools as it


env = gym.make('CarRacing-v0').env
env.reset()
env.render()

all_actions = np.array(
[k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])]
)


q_table = np.zeros([96, env.action_space.n])

#hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1


#for plotting metrics
all_epochs = []
all_penalties = []


for i in range(1,100001):
    state = env.reset()
    epochs,penalties,reward = 0,0,0
    done = False

    while not done:
        if(random.uniform(0,1) < epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state,reward,done,info = env.step(action)

        old_value = q_table[state,action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state,action] = new_value

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training Finished.\n")


