import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# ENVIRONMENT SETTINGS
# -----------------------------

NUM_LANES = 4
MAX_CARS = 20
ACTIONS = 4
EPISODES = 1000
STEPS_PER_EPISODE = 50

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Q-table (3x3x3x3 states × 4 actions)
Q = np.zeros((3,3,3,3,ACTIONS))


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def discretize(cars):
    return tuple(min(c // 7, 2) for c in cars)

def choose_action(state):
    if random.uniform(0,1) < epsilon:
        return random.randint(0, ACTIONS-1)
    return np.argmax(Q[state])

def simulate_step(cars, action):
    # Cars leave from green lane
    cars[action] = max(0, cars[action] - random.randint(3,6))

    # New cars arrive randomly
    for i in range(NUM_LANES):
        cars[i] += random.randint(0,3)
        cars[i] = min(cars[i], MAX_CARS)

    reward = -sum(cars)
    return cars, reward


# -----------------------------
# TRAIN RL AGENT
# -----------------------------

rl_rewards = []

for episode in range(EPISODES):
    cars = [random.randint(0,10) for _ in range(NUM_LANES)]
    total_reward = 0

    for step in range(STEPS_PER_EPISODE):
        state = discretize(cars)
        action = choose_action(state)

        new_cars, reward = simulate_step(cars.copy(), action)
        new_state = discretize(new_cars)

        # Q-learning update
        Q[state][action] = Q[state][action] + alpha * (
            reward + gamma * np.max(Q[new_state]) - Q[state][action]
        )

        cars = new_cars
        total_reward += reward

    rl_rewards.append(total_reward)

print("Training Complete!")


# -----------------------------
# FIXED TIMER SIMULATION
# -----------------------------

def fixed_timer_simulation():
    cars = [random.randint(0,10) for _ in range(NUM_LANES)]
    total_reward = 0

    for step in range(STEPS_PER_EPISODE):
        action = step % 4  # rotate signals
        cars, reward = simulate_step(cars, action)
        total_reward += reward

    return total_reward


# -----------------------------
# TEST RL VS FIXED
# -----------------------------

def test_rl():
    cars = [random.randint(0,10) for _ in range(NUM_LANES)]
    total_reward = 0

    for step in range(STEPS_PER_EPISODE):
        state = discretize(cars)
        action = np.argmax(Q[state])  # best learned action
        cars, reward = simulate_step(cars, action)
        total_reward += reward

    return total_reward


rl_test = []
fixed_test = []

for _ in range(200):
    rl_test.append(test_rl())
    fixed_test.append(fixed_timer_simulation())

print("Average RL Reward:", np.mean(rl_test))
print("Average Fixed Timer Reward:", np.mean(fixed_test))


# -----------------------------
# PLOTS
# -----------------------------

plt.figure()
plt.plot(rl_rewards)
plt.title("RL Training Performance")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.show()

plt.figure()
plt.bar(["RL", "Fixed Timer"], 
        [np.mean(rl_test), np.mean(fixed_test)])
plt.title("RL vs Fixed Timer Comparison")
plt.ylabel("Average Reward")
plt.show()
