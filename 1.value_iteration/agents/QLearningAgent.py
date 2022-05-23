
import numpy as np

# Discrete action spaces and observation spaces.
class QLearningAgent:
    def __init__(self, env, epsilon=0.15, alpha=0.15, gamma=0.95, epsilon_decay_generator=None):
        self.type = 'Q-Learning'
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.episodes = 0
        self.epsilons = []
        self.episodic_rewards = []
        self.episodic_total_steps = []

        if(epsilon_decay_generator is None):
            self.epsilon_generator = lambda  epsilon, episode, max_episodes: 0.99 * epsilon
        else:
            self.epsilon_generator = epsilon_decay_generator


    def get_optimal_action(self, state):
        return np.argmax(self.Q_table[state]) + 1

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state]) + 1
    
    def learn(self, state, action, reward, next_state):
        self.Q_table[state, action - 1] =   self.alpha * (reward + self.gamma * np.max(self.Q_table[next_state])) + self.Q_table[state, action - 1] * (1 - self.alpha) 

    def run_episode(self):
        state = self.env.reset()
        total_reward = 0
        steps = 0
        while True:
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.learn(state, action, reward, next_state)
            state = next_state
            steps += 1
            if done:
                break
        self.episodes += 1
        self.episodic_rewards.append(total_reward)
        self.episodic_total_steps.append(steps)
        return self.Q_table
    
    def train(self, episodes):
        for i in range(episodes):
            self.run_episode()
            self.epsilon = self.epsilon_generator(self.epsilon, i + 1, episodes)
            self.epsilons.append(self.epsilon)
            