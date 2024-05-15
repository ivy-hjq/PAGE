from gym.envs.my_env.FL.utils.utils_general import *
from gym.envs.my_env.FL.utils.utils_methods import *
class FL_synthetic():
    def __init__(self,path,data_path,device):
        self.device = device
        # Dataset initialization
        ###
        [alpha, beta, theta, iid_sol, iid_data, name_prefix] = [0.0, 1.0, 0.0, True , False , 'syn_alpha-0_beta-1_theta0_ddpgtest']

        n_dim = 30
        n_clnt= 100
        n_cls = 30
        avg_data = 375
        split_ratio=0.7
        split_ratio_global=0.8
        self.path = path
        data_path=data_path
        self.data_obj = DatasetSynthetic(alpha=alpha, beta=beta, theta=theta, iid_sol=iid_sol, iid_data=iid_data, n_dim=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data, split_ratio=split_ratio, split_ratio_global=split_ratio_global, name_prefix=name_prefix,data_path=data_path)
        
        self.penalty=1e-5
        ###
        model_name         = 'Linear' # Model type
        self.com_amount         = 1000
        self.save_period        = 20
        self.weight_decay       = 1e-5
        self.batch_size         = 5
        self.act_prob           = 1
        self.lr_decay_per_round = 1
        self.epoch              = 20
        self.learning_rate      = .1
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
        self.model_func = lambda : client_model(model_name, [n_dim, n_cls])
        self.init_model = self.model_func()

        # Initalise the model for all methods
        with torch.no_grad():
            self.init_model.fc.weight = torch.nn.parameter.Parameter(torch.zeros(n_cls,n_dim))
            self.init_model.fc.bias = torch.nn.parameter.Parameter(torch.zeros(n_cls))

        if not os.path.exists('%s/%s/' %(self.path, self.data_obj.name)):
            print("New directory!")
            os.mkdir('%s/%s/' %(self.path, self.data_obj.name))
            torch.save(self.init_model.state_dict(), '%s/%s/%s_init_mdl.pt' %(self.path, self.data_obj.name, model_name))
        else:
            # Load model
            self.init_model.load_state_dict(torch.load('%s/%s/%s_init_mdl.pt' %(self.path, self.data_obj.name, model_name)))    

    def train(self,action,s_action,com_count):
        state_client,state_server,reward_client, reward_server=train_Fed(self,action,s_action,com_count)
        return state_client,state_server,reward_client, reward_server
    def reset(self):
        state=reset_Fed(self)
        return state
    
