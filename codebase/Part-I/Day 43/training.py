# IMPORTING LIBRARIES

import os
import numpy as np
import random as rn
import environment
import brain
import dqn

# SETTING SEEDS

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING PARAMETERS

epsilon = .3
number_actions = 5
direction_boundry = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

brain = brain.Brain(learning_rate=0.00001, number_actions=number_actions)

dqn = dqn.DQN(max_memory=max_memory, discount=0.9)

train = True

env.train = train
model = brain.model
if(env.train):
    for epoch in range(1, number_epochs):
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0,12)
        env.reset(new_month=new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        while((not game_over) and timestep <= 5*30*24*60):
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if(action - direction_boundry < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundry) * temperature_step
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if(action - direction_boundry < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundry) * temperature_step
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            total_reward += reward
            dqn.remember(transition=([current_state, action, reward, next_state]), game_over)
            inputs, targets = dqn.get_batch(model, batch_size=batch_size)
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state

        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))

        model.save("model.h5")