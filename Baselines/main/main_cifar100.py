import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path,os.pardir))
sys.path.insert(0,parent_dir_path)

from utils.utils_general import *
from utils.utils_methods_ import *
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" )

n_client = 100
batch_size = 32
name='CIFAR100'
data_path='Data/Raw/cifar100'
### test_client_number: the number of test client for the method of FedReconstuction, please set test_client_number=0 for other methods.
data_obj = ImagenetObjectCrop_noniid(dataset='CIFAR100', data_path=data_path, n_client=n_client, rule='hetero', unbalanced_sgm=0, rule_arg=0.3,split_ratio=0.7,batch_size=batch_size,test_client_number=0,test_recon_ratio=0.7*0.5)
n_cls = 100
###
model_name        = 'Resnet18'
com_amount         = 1000
save_period        = 50
weight_decay       = 1e-4
act_prob           = 1
lr_decay_per_round = 1
epoch              = 5
learning_rate      = 0.2
print_per          = 20
method_name='FedAvg'
data_obj.name+='_%s_%s_lr%f'%(method_name,model_name,learning_rate)
# Model function
model_func = lambda : client_model(model_name)
# Initalise the model for all methods or load it from a saved initial model
init_model = model_func()
if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
    print("New directory!")
    os.mkdir('Output/%s/' %(data_obj.name))
    torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))    
    
# Methods
if method_name=='FedAvg':
    print('FedAvg')

    [fed_mdls_all_FedAvg,
    tst_perf_all_FedAvg] = train_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name=='FedDyn':       
    print('FedDyn')

    alpha_coef = 1e-2
    [fed_mdls_all_FedFyn, tst_perf_all_FedFyn,
    fed_mdls_cld_FedFyn] = train_FedDyn(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                        model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                        save_period=save_period, lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name=='SCAFFOLD':
    print('SCAFFOLD')

    [fed_mdls_all_SCAFFOLD,
    tst_perf_all_SCAFFOLD] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                            batch_size=batch_size, epoch=epoch, com_amount=com_amount,
                                            print_per=print_per, weight_decay=weight_decay, model_func=model_func,
                                            init_model=init_model, save_period=save_period, lr_decay_per_round=lr_decay_per_round, device=device)
    
elif method_name == 'FedProx':
    print('FedProx')

    mu = 1e-4
    [fed_mdls_all_FedProx, 
    tst_perf_all_FedProx] = train_FedProx(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        mu=mu, lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name == 'pFedMe':
    print('pFedMe')
    
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
    
    mu=0.1 # lambda
    plocal_steps=20
    [fed_mdls_all_FedAvg, 
    tst_perf_all_FedAvg] = train_ditto(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, plocal_steps=plocal_steps, mu=mu,
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
    
    head = copy.deepcopy(init_model.model.fc)
    init_model.model.fc = nn.Identity()
    init_model = BaseHeadSplit(init_model,head) 
    [fed_mdls_all_FedAvg, 
    tst_perf_all_FedAvg] = train_fedrod(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, num_classes=n_cls,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, device=device)

elif method_name =='FedRecon':
    print('FedRecon')
    
    head = copy.deepcopy(init_model.model.fc)
    init_model.model.fc =nn.Identity()
    init_model = BaseHeadSplit(init_model,head)
    recon_epochs=2
    pers_epochs=18
    recon_lr=0.2
    pers_lr=0.2
    s_lr = 0.1
    n_test_client=25
    supset_ratio=0.1
    [fed_mdls_all_FedAvg, 
    tst_perf_all_FedAvg] = train_fedrecon(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        recon_epochs=recon_epochs, pers_epochs=pers_epochs,recon_lr=recon_lr,pers_lr=pers_lr, s_lr=s_lr, n_test_client=n_test_client,supset_ratio=supset_ratio,com_amount=com_amount, print_per=print_per,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, device=device)
else:
    print('method_name is wrong!')
    