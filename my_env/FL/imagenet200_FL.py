from gym.envs.my_env.FL.utils.utils_general import *
from gym.envs.my_env.FL.utils.utils_methods_ import *
import os

class FL_imagenet200():
    def __init__(self,path,data_path,device):
        n_clnt = 100
        self.device = device
        self.batch_size = 128
        data_path=data_path
        self.data_obj = ImagenetObjectCrop_noniid(dataset='Imagenet200', data_path=data_path, n_client=n_clnt, rule='hetero', unbalanced_sgm=0, rule_arg=0.3,split_ratio=0.7,batch_size=self.batch_size)
        self.path= path
        self.penalty=0
        ###
        model_name         = 'Resnet18_200' # Model type
        self.com_amount         = 500
        self.save_period        = 20
        self.weight_decay       = 1e-4
        self.batch_size         = 128
        self.act_prob           = 1
        self.lr_decay_per_round = 1
        self.epoch              = 20
        self.learning_rate      = 0.1
        self.print_per          = 5
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
 
