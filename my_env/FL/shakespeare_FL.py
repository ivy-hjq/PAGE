from gym.envs.my_env.FL.utils.utils_general import *
from gym.envs.my_env.FL.utils.utils_methods import *

import os

class FL_shakespeare():
    def __init__(self,path,data_path,device):
        self.device = device
        # Dataset initialization
        # For Shakespeare experiments
        # First generate dataset using LEAF Framework and set storage_path to the data folder
        storage_path = data_path
        self.path = path
        n_clnt = 100
        #     - In IID use

        # name = 'shakepeare'
        # data_obj = ShakespeareObjectCrop(storage_path, dataset_prefix)

        #     - In non-IID use
        name = 'shakepeare_nonIID'
        dataset_prefix='shakespeare'
        self.data_obj = ShakespeareObjectCrop_noniid(storage_path, dataset_prefix)
        #########

        
        self.penalty=0
        ###
        model_name         = 'shakespeare' # Model type
        self.com_amount         = 1000
        self.save_period        = 20
        self.weight_decay       = 1e-4
        self.batch_size         = 32
        self.act_prob           = 1
        self.lr_decay_per_round = 1
        self.epoch              = 5
        self.learning_rate      = 1
        self.print_per          = 20
        self.rand_seed          = 0

        self.n_save_instances = int(self.com_amount / self.save_period)
        self.fed_mdls_all = list(range(self.n_save_instances))
        self.tst_perf_all = np.zeros((self.com_amount, 2))
        
        self.tst_perf_global=[[] for i in range(n_clnt)]
        self.trn_perf_clnt=[[] for i in range(n_clnt)]
        self.tst_perf_clnt=[[] for i in range(n_clnt)]
        self.tst_perf_clnt_=[[] for i in range(n_clnt)]
        # Model function
        self.model_func = lambda : client_model(model_name)
        # Initalise the model for all methods or load it from a saved initial model
        self.init_model = self.model_func()
        if not os.path.exists('%s/%s/%s_init_mdl.pt' %(self.path, self.data_obj.name, model_name)):
            print("New directory!")
            os.mkdir('%s/%s/' %(self.path, self.data_obj.name))
            torch.save(self.init_model.state_dict(), '%s/%s/%s_init_mdl.pt' %(self.path, self.data_obj.name, model_name))
        else:
            # Load model
            self.init_model.load_state_dict(torch.load('%s/%s/%s_init_mdl.pt' %(self.path, self.data_obj.name, model_name)))    
            
        # Methods    
    def train(self,action,s_action,com_count):
        state_client,state_server,reward_client, reward_server=train_Fed(self,action,s_action,com_count)
        return state_client,state_server,reward_client, reward_server
    
    def reset(self):
        state=reset_Fed(self)
        return state
 