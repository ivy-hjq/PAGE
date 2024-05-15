import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path,os.pardir))
sys.path.insert(0,parent_dir_path)

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from utils.model import (Actor, Critic)
from utils.memory import SequentialMemory
from utils.random_process import OrnsteinUhlenbeckProcess
from utils.util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, nb_agents,args,hidden1,hidden2,rate,prate,bsize,lr_max,lr_min,epoch_max, is_lr, is_epoch, is_p):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epoch_max = epoch_max
        self.scale = int(self.epoch_max/self.lr_max)
        self.is_lr = is_lr
        self.is_epoch = is_epoch
        self.is_p = is_p  
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.nb_agents = nb_agents
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':self.hidden1, 
            'hidden2':self.hidden2, 
            'init_w':args.init_w
        }
        self.actor_optim_list=[]
        self.critic_optim_list=[]

        

        self.actor_list = [Actor(self.nb_states, self.nb_actions, **net_cfg) for i in range (self.nb_agents)]
        self.actor_target_list = [Actor(self.nb_states, self.nb_actions, **net_cfg) for i in  range (self.nb_agents)]
        
        self.critic_list = [Critic(self.nb_states, self.nb_actions, **net_cfg) for i in range (self.nb_agents)]
        self.critic_target_list = [Critic(self.nb_states, self.nb_actions, **net_cfg) for i in range (self.nb_agents)]
        
        
        for i in range (self.nb_agents):
            self.actor_optim_list.append(Adam(self.actor_list[i].parameters(), lr=prate))
            self.critic_optim_list.append(Adam(self.critic_list[i].parameters(), lr=rate))
            hard_update(self.actor_target_list[i], self.actor_list[i]) # Make sure target is with the same weight
            hard_update(self.critic_target_list[i], self.critic_list[i])
        
        #Create replay buffer
        self.memory = [SequentialMemory(limit=args.rmsize, window_length=args.window_length) for i in range (self.nb_agents)]
        self.random_process = [OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma) for i in range (self.nb_agents)]
        
        
        # Hyper-parameters
        self.batch_size = bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon 

        # 
        self.epsilon = [1.0 for i in range(self.nb_agents)]
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: 
            self.cuda()

    def update_policy(self):
        p_loss=[]
        v_loss=[]
        for i in range(self.nb_agents):
            # Sample batch
            state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory[i].sample_and_split(self.batch_size)
            


            # Prepare for the target q batch
            with torch.no_grad():
                next_q_values = self.critic_target_list[i]([
                    to_tensor(next_state_batch),
                    self.actor_target_list[i](to_tensor(next_state_batch)),
                ])
            # next_q_values.volatile=False

            target_q_batch = to_tensor(reward_batch) + \
                self.discount*to_tensor(terminal_batch.astype(np.float64))*next_q_values
            

            # Critic update
            self.critic_list[i].zero_grad()

            q_batch = self.critic_list[i]([ to_tensor(state_batch), to_tensor(action_batch) ])
    

            value_loss = criterion(q_batch, target_q_batch)
            value_loss.backward()
            self.critic_optim_list[i].step()

            # Actor update
            self.actor_list[i].zero_grad()

            policy_loss = -self.critic_list[i]([
                to_tensor(state_batch),
                self.actor_list[i](to_tensor(state_batch))
            ])
            
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim_list[i].step()

            # Target update
            soft_update(self.actor_target_list[i], self.actor_list[i], self.tau)
            soft_update(self.critic_target_list[i], self.critic_list[i], self.tau)
            p_loss.append(policy_loss)
            v_loss.append(value_loss)
        return p_loss,v_loss

    def eval(self):
        for i in range(self.nb_agents):
            self.actor_list[i].eval()
            self.actor_target_list[i].eval()
            self.critic_list[i].eval()
            self.critic_target_list[i].eval()

    def cuda(self):
        for i in range(self.nb_agents):                
            self.actor_list[i].cuda()
            self.actor_target_list[i].cuda()
            self.critic_list[i].cuda()
            self.critic_target_list[i].cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            for i in range(self.nb_agents):
                self.memory[i].append(self.s_t[i], self.a_t[i], r_t[i], done)   
            self.s_t = s_t1
   
    def load_observe(self,state,action,reward,done):
        for i in range(self.nb_agents):
            self.memory[i].append(state[i], action[i], reward[i], done)

    def random_a(self):
        np.random.seed(None)
        action = np.random.uniform(self.lr_max,self.lr_min,(self.nb_agents,self.nb_actions))
        
        for i in range(self.nb_agents):
            
            action[i][-1]*=self.scale
            if action[i][-1]<1:
                action[i][-1]=1
        if not self.is_lr:
            for i in range(self.nb_agents):
                action[i][0]=self.lr_max
        if not self.is_epoch:
            for i in range(self.nb_agents):
                action[i][-1]=self.epoch_max
                
        self.a_t = action
        return action

    def select_a(self, s_t, decay_epsilon=True):
        np.random.seed(None)
        action=[]
        for i in range(self.nb_agents):
            a = to_numpy(
                self.actor_list[i](to_tensor(np.array([s_t[i]])))
            ).squeeze()
            a += self.is_training*max(self.epsilon[i], 0)*self.random_process[i].sample()
            

            a=(a+1)/2
            a = np.clip(a, self.lr_min, self.lr_max)
            a[-1]*=self.scale
            if a[-1]<1:
                a[-1]=1

            if decay_epsilon:
                self.epsilon[i] -= self.depsilon
            
            if not self.is_lr:
                a[0] = self.lr_max
            
            if not self.is_epoch:
                a[-1] = self.epoch_max
            action.append(a)
            
        self.a_t=action
        return action

    def random_sa(self):
        np.random.seed(None)
        action = np.random.uniform(0,1,(self.nb_agents,self.nb_actions))
        for i in range(self.nb_agents):
            action[i]=action[i]/sum(action[i])#normalize sum to 1
        if not self.is_p:
            action = np.full((self.nb_agents,self.nb_actions), 1/self.nb_actions)
        self.a_t = action
        return action

    def select_sa(self, s_t, decay_epsilon=True):
        np.random.seed(None)
        action=[]
        for i in range(self.nb_agents):
            a = to_numpy(
                self.actor_list[i](to_tensor(np.array(s_t)))
            ).squeeze()
            

            a=(a+1)/2 
            a=a/sum(a)
            a=np.clip(a, 0, 1)
            if decay_epsilon:
                self.epsilon[i] -= self.depsilon 
            action.append(a)
        if not self.is_p:
            action =  np.full((self.nb_agents,self.nb_actions), 1/self.nb_actions)   
        self.a_t=action
        return action

    def reset(self, obs):
        self.s_t=[]
        for i in range(self.nb_agents):
            self.s_t.append(obs[i])
            self.random_process[i].reset_states()

    def load_weights(self, output):
        if output is None: return
        for i in range(self.nb_agents):
            self.actor_list[i].load_state_dict(
                torch.load('{}/actor{}.pkl'.format(output,i))
            )
            self.actor_target_list[i].load_state_dict(
                torch.load('{}/actor_target{}.pkl'.format(output,i))
            )

            self.critic_list[i].load_state_dict(
                torch.load('{}/critic{}.pkl'.format(output,i))
            )
            self.critic_target_list[i].load_state_dict(
                torch.load('{}/critic_target{}.pkl'.format(output,i))
            )
    def load_i_weights(self, output,step):
        if output is None: return
        for i in range(self.nb_agents):
            self.actor_list[i].load_state_dict(
                torch.load('{}/{}actor{}.pkl'.format(output,step,i))
            )
            self.actor_target_list[i].load_state_dict(
                torch.load('{}/{}actor_target{}.pkl'.format(output,step,i))
            )

            self.critic_list[i].load_state_dict(
                torch.load('{}/{}critic{}.pkl'.format(output,step,i))
            )
            self.critic_target_list[i].load_state_dict(
                torch.load('{}/{}critic_target{}.pkl'.format(output,step,i))
            )



    def save_model(self,output):
        for i in range(self.nb_agents):
            torch.save(
                self.actor_list[i].state_dict(),
                '{}/actor{}.pkl'.format(output,i)
            )
            torch.save(
                 self.actor_target_list[i].state_dict(),
                '{}/actor_target{}.pkl'.format(output,i)
            )
            torch.save(
                self.critic_list[i].state_dict(),
                '{}/critic{}.pkl'.format(output,i)
            )
            torch.save(
                self.critic_target_list[i].state_dict(),
                '{}/critic_target{}.pkl'.format(output,i)
            )
    def save_inter_model(self,output,step):
        for i in range(self.nb_agents):
            torch.save(
                self.actor_list[i].state_dict(),
                '{}/{}actor{}.pkl'.format(output,step,i)
            )
            torch.save(
                 self.actor_target_list[i].state_dict(),
                '{}/{}actor_target{}.pkl'.format(output,step,i)
            )
            torch.save(
                self.critic_list[i].state_dict(),
                '{}/{}critic{}.pkl'.format(output,step,i)
            )
            torch.save(
                self.critic_target_list[i].state_dict(),
                '{}/{}critic_target{}.pkl'.format(output,step,i)
            )
    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

