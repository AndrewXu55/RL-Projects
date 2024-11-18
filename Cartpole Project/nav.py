# Import necessary libraries
import numpy as np  # For numerical operations
import gym  # For creating the CartPole environment
import random  # For random action selection
import tensorflow as tf  # For building the neural network model
from tensorflow.keras.layers import Input  # For defining input layers in Keras
from collections import deque  # For storing experiences in memory
import cv2  # For displaying environment frames
import matplotlib
import matplotlib.pyplot as plt  # For plotting rewards
from threading import Thread  # For parallel plotting during training

matplotlib.use('Agg')  # Use non-interactive backend

# Define the DQLAgent class, representing the deep Q-learning agent
class DQLAgent:
    def __init__(self, state_size, action_size):
        """
        Initialize the DQLAgent.

        Parameters:
        - state_size: Integer, the size of the state space
        - action_size: Integer, the size of the action space
        """
        self.state_size = state_size  # Dimensionality of input (state)
        self.action_size = action_size  # Number of possible actions
        self.memory = deque(maxlen=20000)  # Store past experiences with a maximum length of 20,000
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate (start with high exploration)
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon to gradually reduce exploration
        self.learning_rate = 0.001  # Learning rate for the neural network
        self.model = self._build_model()  # Build the Q-network model

    def _build_model(self):
        """
        Build the neural network for Q-learning.

        The network has an input layer followed by two dense layers with 24 neurons each,
        using ReLU activation, and an output layer with 'linear' activation for Q-values.

        Returns:
        - model: Compiled Keras model
        """
        model = tf.keras.Sequential()  # Initialize a sequential model
        model.add(Input(shape=(self.state_size,)))  # Define input layer with state size
        model.add(tf.keras.layers.Dense(24, activation='relu'))  # First hidden layer with 24 neurons
        model.add(tf.keras.layers.Dense(24, activation='relu'))  # Second hidden layer with 24 neurons
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))  # Output layer for action Q-values
        # Compile the model with mean squared error loss and Adam optimizer
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience tuple in memory.

        Parameters:
        - state: Current state
        - action: Action taken in the current state
        - reward: Reward received after taking the action
        - next_state: Resulting state after taking the action
        - done: Boolean, whether the episode ended after this action
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action based on epsilon-greedy policy.

        Parameters:
        - state: Current state, used to predict action

        Returns:
        - action: Chosen action (either random or predicted by model)
        """
        if np.random.rand() <= self.epsilon:  # With probability epsilon, select a random action
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)  # Predict Q-values for the current state
        return np.argmax(act_values[0])  # Select action with the highest Q-value

    def replay(self, batch_size):
        """
        Train the Q-network using experiences from memory.

        Parameters:
        - batch_size: Number of experiences to sample from memory
        """
        if len(self.memory) < batch_size:  # Check if enough samples are in memory
            return
        minibatch = random.sample(self.memory, batch_size)  # Sample a random minibatch
        for state, action, reward, next_state, done in minibatch:
            # Set the target Q-value
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            # Predict current Q-values and update only the Q-value for the selected action
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            # Fit the model for one training step on the modified Q-values
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Reduce epsilon to decrease exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Function to plot the moving average of rewards
def plot_moving_average(rewards, window_size=25):
    """
    Plot the moving average of rewards to visualize agent's performance.

    Parameters:
    - rewards: List of total rewards for each episode
    - window_size: Integer, size of the moving average window
    """
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size - 1, len(rewards)), smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Total Reward')
    plt.title('Smoothed Total Reward over Episodes')
    plt.grid()
    plt.savefig('plot.png')


# Function to train the agent in the environment
def train_agent(agent, env, episodes=1000):
    """
    Train the DQLAgent in a given environment.

    Parameters:
    - agent: DQLAgent instance to be trained
    - env: Gym environment instance
    - episodes: Total number of episodes for training

    Returns:
    - rewards: List of total rewards for each episode
    """
    state_size = env.observation_space.shape[0]  # Get the dimension of the state space
    batch_size = 32  # Size of minibatch for replay
    rewards = []  # List to store rewards for each episode

    for e in range(episodes):  # Loop over episodes
        state = env.reset()[0]  # Reset the environment
        state = np.reshape(state, [1, state_size])  # Reshape state for model input
        total_reward = 0  # Initialize total reward for the episode
        done = False  # Flag to check if the episode is done

        while not done:
            action = agent.act(state)  # Select an action based on current policy
            next_state, reward, done, _, info = env.step(action)  # Take action in the environment
            reward = reward if not done else -10  # Penalize end of episode with -10
            next_state = np.reshape(next_state, [1, state_size])  # Reshape next state for model input
            agent.remember(state, action, reward, next_state, done)  # Store experience
            state = next_state  # Move to next state
            total_reward += reward  # Accumulate reward

            # Display the frame every 20th episode
            if e % 20 == 0:
                frame = env.render()  # Capture the frame in RGB format
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
                    position = state[0][0]  # Cart position
                    angle = state[0][2]  # Pole angle
                    text = f"Position: {position:.2f}, Angle: {angle:.2f}, Time: {total_reward}"
                    cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('CartPole', frame_bgr)
                    if cv2.waitKey(100) & 0xFF == ord('q'):  # Break if 'q' key is pressed
                        break

        print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        rewards.append(total_reward)  # Add total reward for the episode to the list

        if len(agent.memory) > batch_size:  # Train the agent with experiences from memory
            agent.replay(batch_size)

        # Plot the moving average every 25 episodes
        if (e + 1) % 25 == 0:
            plot_moving_average(rewards)

    # Final plot of the total rewards over episodes
    plt.plot(range(episodes), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.grid()
    plt.savefig('plot2.png')

    return rewards  # Return the list of rewards


# Initialize the CartPole environment
env = gym.make('CartPole-v1', render_mode='rgb_array')

# Create and train the DQLAgent
agent = DQLAgent(env.observation_space.shape[0], env.action_space.n)
rewards = train_agent(agent, env, episodes=1000)

# Cleanup: close environment and OpenCV windows
env.close()
cv2.destroyAllWindows()