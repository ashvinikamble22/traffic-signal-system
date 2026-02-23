import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Traffic Signal RL", layout="wide")

# -----------------------------
# PARAMETERS
# -----------------------------

NUM_LANES = 4
MAX_CARS = 20
ACTIONS = 4
EPISODES = 500
STEPS = 50

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# -----------------------------
# FUNCTIONS
# -----------------------------

def discretize(cars):
    return tuple(min(c // 7, 2) for c in cars)

def simulate_step(cars, action):
    cars[action] = max(0, cars[action] - random.randint(3,6))
    for i in range(NUM_LANES):
        cars[i] += random.randint(0,3)
        cars[i] = min(cars[i], MAX_CARS)
    reward = -sum(cars)
    return cars, reward

def train_rl():
    Q = np.zeros((3,3,3,3,ACTIONS))
    rewards = []

    for ep in range(EPISODES):
        cars = [random.randint(0,10) for _ in range(NUM_LANES)]
        total_reward = 0

        for _ in range(STEPS):
            state = discretize(cars)

            if random.uniform(0,1) < epsilon:
                action = random.randint(0, ACTIONS-1)
            else:
                action = np.argmax(Q[state])

            new_cars, reward = simulate_step(cars.copy(), action)
            new_state = discretize(new_cars)

            Q[state][action] += alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state][action]
            )

            cars = new_cars
            total_reward += reward

        rewards.append(total_reward)

    return Q, rewards

def test_model(Q):
    cars = [random.randint(0,10) for _ in range(NUM_LANES)]
    total_reward = 0

    for _ in range(STEPS):
        state = discretize(cars)
        action = np.argmax(Q[state])
        cars, reward = simulate_step(cars, action)
        total_reward += reward

    return total_reward

def fixed_timer():
    cars = [random.randint(0,10) for _ in range(NUM_LANES)]
    total_reward = 0

    for step in range(STEPS):
        action = step % 4
        cars, reward = simulate_step(cars, action)
        total_reward += reward

    return total_reward


# -----------------------------
# UI
# -----------------------------

st.title("🚦 Smart Traffic Signal Using Reinforcement Learning")

st.write("This system learns optimal signal timing to reduce traffic congestion.")

if st.button("Train RL Model"):
    with st.spinner("Training in progress..."):
        Q, training_rewards = train_rl()
        st.session_state["Q"] = Q

    st.success("Training Complete!")

    fig1 = plt.figure()
    plt.plot(training_rewards)
    plt.title("Training Performance")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    st.pyplot(fig1)


if "Q" in st.session_state:
    if st.button("Compare RL vs Fixed Timer"):
        rl_result = np.mean([test_model(st.session_state["Q"]) for _ in range(50)])
        fixed_result = np.mean([fixed_timer() for _ in range(50)])

        fig2 = plt.figure()
        plt.bar(["RL", "Fixed Timer"], [rl_result, fixed_result])
        plt.ylabel("Average Reward")
        plt.title("Performance Comparison")
        st.pyplot(fig2)

        st.write("RL Average Reward:", rl_result)
        st.write("Fixed Timer Average Reward:", fixed_result)
