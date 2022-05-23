
import numpy as np

# We use the On-policy first-visit MC Control with epsilon soft policy
class OnPolicyMonteCarloAgent:

    def __init__(self, env, epsilon=0.15, gamma=0.95, epsilon_decay_generator=None):
        self.type = 'On-policy Monte Carlo'
        self.env = env
        self.gamma = gamma
        self.QTable = np.zeros((env.observation_space.n, env.action_space.n)) # Q Table
        self.Returns = np.zeros((env.observation_space.n, env.action_space.n)) # Returns matrix
        self.NTable = np.zeros((env.observation_space.n, env.action_space.n)) # Number of times state-action pair has been visited; used for averaging
        
        # Action probabilities for each action equal probability distribution over all actions
        self.action_probabilities = np.ones((env.observation_space.n, env.action_space.n))/ env.action_space.n
        self.epsilons = []
        self.episodic_rewards = []
        self.episodes = 0
        self.episodic_total_steps = []
        self.epsilon = epsilon

        if(epsilon_decay_generator is None):
            self.epsilon_generator = lambda  epsilon, episode, max_episodes: 0.99 * epsilon
        else:
            self.epsilon_generator = epsilon_decay_generator


    # Return optimal action for state
    def get_optimal_action(self, state):
        return np.argmax(self.QTable[state]) + 1

    # Generate an episode using the policy
    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            # get best action from policy matrix
            action = np.random.choice(np.arange(self.env.action_space.n) + 1, p=self.action_probabilities[state])
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            episode.append((state, action, reward, next_state))
            state = next_state
            steps += 1
        self.episodic_rewards.append(total_reward)
        self.episodic_total_steps.append(steps)
        return episode

    # Reference from Richard Sutton's book
    def update_Q(self, episode):
        G = 0 # G is discounted cumulative reward
        visited = []

        # loop episode in reverse
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward, next_state = episode[i]
            G = self.gamma * G + reward
            if (state, action) not in visited:
                # Append G to Returns(State, action) list
                self.Returns[(state, action - 1)] += G
                
                # update N to number of times state-action pair has been visited
                self.NTable[state, action - 1] += 1
                
                # Update Q to average of returns
                self.QTable[state, action - 1] = self.Returns[(state, action - 1)] / self.NTable[state, action - 1]
                

                # Optimal action is the action that maximizes the Q value 
                optimal_action = np.argmax(self.QTable[state]) + 1
                # Update action probabilities
                for i in range(self.env.action_space.n):
                    if i + 1 == optimal_action:
                        self.action_probabilities[state][i] = 1 - self.epsilon + self.epsilon / self.env.action_space.n
                    else:
                        self.action_probabilities[state][i] = self.epsilon / self.env.action_space.n
    
    def train(self, episodes=500):
        # Run 15000 episodes
        for i in range(episodes):
            # Generate an episode
            episode = self.generate_episode()
            self.update_Q(episode)
            # Update epsilon
            # use epsilon generator pass old value of epsilon
            self.epsilon = self.epsilon_generator(self.epsilon, i + 1, episodes)
            self.epsilons.append(self.epsilon)
            self.episodes += 1
