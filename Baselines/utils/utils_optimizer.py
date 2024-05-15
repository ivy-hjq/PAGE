import torch
from torch.optim import Optimizer 
            

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr, lamda, mu):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params,defaults)
        
    def step(self, local_model, device):
        param_group = None
        weight_update = local_model.copy()
        for param_group in self.param_groups:
            for param, localweight in zip(param_group['params'],weight_update):
                localweight = localweight.to(device)
                # approximate local model
                param.data = param.data - param_group['lr'] * (param.grad.data+param_group['lamda']*(param.data-localweight.data)+param_group['mu']*param.data)
        return param_group['params']   
    
class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr, mu):
        default = dict(lr=lr,mu=mu)
        super().__init__(params,default)
        
    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])      
  
class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)               