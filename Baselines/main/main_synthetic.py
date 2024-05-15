import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path,os.pardir))
sys.path.insert(0,parent_dir_path)

from utils.utils_general import *
from utils.utils_methods import *
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" )
# Dataset initialization
###
[alpha, beta, theta, iid_sol, iid_data, name_prefix] = [0.0, 1.0, 0.0, True , False , 'syn_alpha-0_beta-1_theta0']

n_dim = 30
n_clnt= 100
n_cls = 30
avg_data = 375
split_ratio=0.7
split_ratio_global=0.8

data_obj = DatasetSynthetic(alpha=alpha, beta=beta, theta=theta, iid_sol=iid_sol, iid_data=iid_data, n_dim=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data, split_ratio=split_ratio, split_ratio_global=split_ratio_global,name_prefix=name_prefix)


###
model_name         = 'Linear' # Model type
com_amount         = 500
save_period        = 20
weight_decay       = 1e-5
batch_size         = 5
act_prob           = 1
lr_decay_per_round = 1
epoch              = 20
learning_rate      = .1
print_per          = 20
method_name = 'FedAvg'
data_obj.name+='_%s_test'%method_name
# Model function
model_func = lambda : client_model(model_name, [n_dim, n_cls])
init_model = model_func()

# Initalise the model for all methods
with torch.no_grad():
    init_model.fc.weight = torch.nn.parameter.Parameter(torch.zeros(n_cls,n_dim))
    init_model.fc.bias = torch.nn.parameter.Parameter(torch.zeros(n_cls))

if not os.path.exists('Output/%s/' %(data_obj.name)):
    os.mkdir('Output/%s/' %(data_obj.name))


# Methods  
if method_name == 'FedAvg':
    print('FedAvg')

    [fed_mdls_all_FedAvg, 
    tst_perf_all_FedAvg] = train_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name == 'FedDyn':
    print('FedDyn')
    
    alpha_coef = 1e-2
    [fed_mdls_all_FedFyn, tst_perf_all_FedFyn,
    fed_mdls_cld_FedFyn] = train_FedDyn(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                        model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                        save_period=save_period, lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name == 'SCAFFOLD':
    print('SCAFFOLD')
    n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_clnt
    n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
    n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
    print_per_ = print_per*n_iter_per_epoch

    [fed_mdls_all_SCAFFOLD, 
    tst_perf_all_SCAFFOLD] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                            batch_size=batch_size, n_minibatch=n_minibatch, com_amount=com_amount,
                                            print_per=print_per_, weight_decay=weight_decay, model_func=model_func,
                                            init_model=init_model, save_period=save_period, lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name == 'FedProx':
    print('FedProx')

    mu = 1e-4
    # mu = 1e-3

    [fed_mdls_all_FedProx, 
    tst_perf_all_FedProx] = train_FedProx(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        mu=mu, lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name == 'pFedMe':
    print('pFedMe')
    learning_rate = 0.005
    beta=1
    lamda=15
    K=5
    mu=0.005
    [fed_mdls_all_FedAvg, 
    tst_perf_all_FedAvg] = train_pFedme(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, beta=beta, lamda=lamda, K=K,mu=mu,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name == 'Ditto':
    print('Ditto')
    # mu=0.005
    mu=0.1 # lambda
    plocal_steps=20
    [fed_mdls_all_FedAvg, 
    tst_perf_all_FedAvg] = train_ditto(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, beta=beta, plocal_steps=plocal_steps, mu=mu,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name =='ALA':
    print('ALA')
    eta=0.2 #ALA learning rate
    rand_percent=80 #ALA data pecent
    layer_idx=1 #ALA layer
    [fed_mdls_all_FedAvg, 
    tst_perf_all_FedAvg] = train_ala(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, eta=eta, rand_percent=rand_percent, layer_idx=layer_idx,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name =='FedRoD':
    print('FedRoD')
    head = copy.deepcopy(init_model.fc)
    init_model.fc = nn.Identity()
    init_model = BaseHeadSplit(init_model,head) 
    [fed_mdls_all_FedAvg, 
    tst_perf_all_FedAvg] = train_fedrod(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, num_classes=n_cls,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, device=device)

else:
    print('method_name is wrong!')
    