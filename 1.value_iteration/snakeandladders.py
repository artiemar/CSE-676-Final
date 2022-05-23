
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from enum import Enum
from PIL import Image
from datetime import datetime

class GameMode(Enum):
    STOCHASTIC = 1,
    DETERMINISTIC = 2

class SnakeAndLadders(gym.Env):

    def __init__(self, snakes = {98: 79, 95: 75, 93: 73, 87: 36,  17: 7, 54: 34, 62: 19, 64: 60  }
    , ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 51: 67, 72:91, 80:99}, max_die_size = 6, max_steps = 100,
     game_mode = GameMode.DETERMINISTIC):
        
        self.action_space = gym.spaces.Discrete(max_die_size, start = 1)
        self.observation_space = gym.spaces.Discrete(max_steps + 1)
        self.ladders_pos = ladders
        self.snakes_pos = snakes
        self.steps = 0
        self.state = 0 # We only care about our current position in this game. Past actions don't affect future states.
        self.maximum_steps = max_steps # maximum number of steps in the game
        self.done = False
        self.game_mode = game_mode
        self.stochastic_policy = 0.76
        self.game_mode_readable = 'Stochastic' if self.game_mode == GameMode.STOCHASTIC else 'Deterministic'

    def step(self, action):
        
        # In a stochastic environment, fulfill agent's action only with a probability of self.stochastic_policy
        if self.game_mode == GameMode.STOCHASTIC:
            if np.random.random() > self.stochastic_policy:
                action = np.random.randint(1, self.action_space.n)
            
                
        # done if we go out of bounds. clip it here to 100.
        self.state = np.clip(self.state + action, 0, self.maximum_steps)
        self.steps += 1

        if self.state == 100:
            reward = 1
            self.done = True
        else:
            reward = -1
            self.done = False


        # Reward of 1 on taking a ladder
        if self.state in self.ladders_pos:
            self.state = self.ladders_pos[self.state]
            reward = 1
        
        # Reward of -1 on ending up in a snake pit
        elif self.state in self.snakes_pos:
            self.state = self.snakes_pos[self.state]
            reward += -1

        if self.steps == self.maximum_steps:
            self.done = True

        return self.state, reward, self.done, {}

    def reset(self):
        self.state = 0
        self.steps = 0
        self.done = False
        return self.state

    def render(self, mode='human'):

        if mode == 'human':
            print("Current state: ", self.state)
            print("Current done: ", self.done)
            print("Current steps: ", self.steps)
            print("\n")


        with open("resources/board.jpg",'rb') as file:
            wallpaper = plt.imread(file, format='jpg')

        fig, ax = plt.subplots()

        ax.imshow(wallpaper, extent=[0, 1, 0, 1])
        ax.set_axis_off()
        with open("resources/agent.png",'rb') as file:
            arr_img = plt.imread(file, format='png')

        imagebox = OffsetImage(arr_img, zoom=0.2)
        imagebox.image.axes = ax

        # get co-ordinates from state
        y = self.state/10
        x = self.state % 10

        x  = np.clip((x - 1)/10  + 0.05 if np.floor(y) % 2 == 0 else (1- x/10 + 0.05),0.05,0.95)
        y = np.clip(np.floor(y if y * 10 % 10 != 0 else (y - 1))/10 + 0.05, 0.05, 0.95)
        
        # clip the borders.
        ab = AnnotationBbox(imagebox, (x , y),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.2,
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=3"))

        ax.add_artist(ab)
        plt.title(f'{self.game_mode_readable}; steps: {self.steps}')
        if(mode == 'human'):
            plt.show()

        buf = None
        if(mode == 'rgb_array'): 
            fig.canvas.draw()
            # returning pillow image, which is technically a wrapper around numpy array.
            buf = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()) 
            plt.close()

        return buf  