#!/usr/bin/env python3 
# state-of-the-art shitcode
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path,os.pardir))
sys.path.insert(0,parent_dir_path)
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import pickle

from utils.normalized_env import NormalizedEnv
from ddpg_dap import DDPG
from utils.util import *


def train(args,num_iterations, num_client, agent, s_agent,env, max_episode_length=None, debug=False):  
    agent.is_training = True
    step = 0
    observation = None
    s_observation = None
    last_reward=np.zeros(num_client)
    while step < num_iterations:
        print("----------------step:",step,"-------------------")
        # reset if it is the start of episode
        if observation is None:
            observation=env.reset()
            s_observation=[]
            s_observation=deepcopy(np.array([observation[num_client-1:-1]]))
            print(s_observation)
            observation = deepcopy(list(observation[0:num_client]))
            for i in range(len(observation)):
                observation[i]=[observation[i],5.0]
            observation=np.array(observation)
            print(observation,s_observation)
            agent.reset(observation)
            s_agent.reset(s_observation)
            print("reset:state.",observation)
            print("reset:s_state.",s_observation)
        
        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_a()
            s_action=s_agent.random_sa()
        else:
            action = agent.select_a(observation)
            s_action=s_agent.select_sa(s_observation)
        print("agent pick action:",action,s_action)
        
        
        # env response with next_observation, reward, terminate_info
        observation2, s_observation2,now_reward,s_reward, done, info = env.step_a(action,s_action,step)

        reward=np.array(now_reward)-last_reward
        last_reward=np.array(now_reward)
        s_observation2 = deepcopy(s_observation2)
        observation2 = deepcopy(observation2)
        if step >= max_episode_length -1:
            done = True
        print("env response with next_observation, reward:", observation2,reward)

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        s_agent.observe(s_reward,s_observation2,done)
       
        
        if step > args.warmup:
            policy_loss,value_loss = agent.update_policy()
            print("agent observe and update policy step:",step)
            print("policy loss and value loss:",policy_loss,value_loss)
        # update 
        step += 1
        observation = deepcopy(observation2)
        s_observation=deepcopy(s_observation2)
        
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='Dap_cifar100-v0', type=str, help='open-ai gym environment, PAGE-v1 shakespear, PAGE-v2 synthetic')
    parser.add_argument('--hidden1', default=40, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=30, type=int, help='hidden num of second fully connect layer')
   
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=35, type=int, help='time without training but only filling the replay memory')
   
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--max_episode_length', default=2000, type=int, help='')
    
    parser.add_argument('--data_path', default='Data/cifar100', type=str, help='')
    parser.add_argument('--output', default='Output/Dap/cifar100', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=1000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=2023, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    parser.add_argument('--cuda', dest='cuda:0', action='store_true') # TODO
    parser.add_argument('--num_clients', default='100', type=int)
    parser.add_argument('--lr_max', default=2e-1, type=float)
    parser.add_argument('--lr_min', default=1e-2, type=float)
    parser.add_argument('--epoch_max', default=20, type=int) 

    args = parser.parse_args()

    args.output = get_output_folder(args.output, args.env)
    env = NormalizedEnv(gym.make(args.env,args.output,args.data_path,args.cuda))

    
    print(env.observation_space.shape)
    print(env.action_space.shape)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    nb_agents= env.nb_agents
    print(nb_states,nb_actions,nb_agents)
    s_nb_actions=nb_agents
    s_nb_states=nb_agents
    s_nb_agents=1
    print(s_nb_states,s_nb_actions,s_nb_agents)
    
    
    agent = DDPG(nb_states, nb_actions, nb_agents,args,hidden1=args.hidden1,hidden2=args.hidden2,bsize=args.bsize,lr_max=args.lr_max, lr_min=args.lr_min, epoch_max=args.epoch_max)
    s_agent=DDPG(s_nb_states,s_nb_actions,s_nb_agents,args,hidden1=args.s_hidden1,hidden2=args.s_hidden2,bsize=args.s_bsize,lr_max=args.lr_max, lr_min=args.lr_min, epoch_max=args.epoch_max)

    
    
    if args.mode == 'train':
        train(args,args.train_iter, args.num_clients,agent, s_agent, env, max_episode_length=args.max_episode_length, debug=args.debug )


    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
