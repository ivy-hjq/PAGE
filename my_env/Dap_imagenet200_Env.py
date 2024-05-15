import math
from typing import Optional, Union

import numpy as np
import pygame
from pygame import gfxdraw

import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.my_env.FL.imagenet200_FL import *

class Dap_imagenet200_Env(gym.Env):
    """
    ### Description

    

    ### Action Space

    
    ### Observation Space

   
    ### Rewards

   
    ### Starting State

   
    ### Episode Termination

   
    ### Arguments

    ```
    gym.make('FLR_lr-v0')
    ```

    No additional arguments are currently supported.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self,output,data_path,device):

        #respents the number of agents, result in the length of action list
        self.nb_agents=100
        self.max_action=np.array([1,20]) 
        self.min_action=np.array([5e-2,1])
        self.max_state=np.array([100.0,100])
        self.min_state=np.array([0.0,0.0])
        

        self.action_space=spaces.Box(
            low=self.min_action,high=self.max_action
        )
        self.observation_space=spaces.Box(
            low=self.min_state,high=self.max_state
        )
        self.FL=FL_imagenet200(output,data_path,device)
        

    
       

    def step_a(self, a, s_a,step):
        #interacte with env
        self.client_state,self.server_state,client_reward,server_reward=self.FL.train(a, s_a, step)
        for i in range(self.nb_agents):
            client_reward[i]=1/client_reward[i]
            self.client_state[i].append(client_reward[i])
            client_reward[i]=self.client_state[i][0]-self.client_state[i][1]
        
        return np.array(self.client_state),np.array(self.server_state),client_reward,server_reward,False,{}
        

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        #reset local model
        self.state=self.FL.reset()
        self.episode_step=0
        print(np.array(self.state))
        return np.array(self.state)
        
    def render(self):

        return
        

    def close(self):

        return

    def _get_observation(self,action):

        return 

    def _get_reward(self):

        return 

    def _get_done(self):

        return done
