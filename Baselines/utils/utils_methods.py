from .utils_libs import *
from .utils_dataset import *
from .utils_models import *
from .utils_general import *

### Methods
def train_FedAvg(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, device, rand_seed=0):
    method_name = 'FedAvg'
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_all = list(range(n_save_instances))
    
    tst_perf_all = np.zeros((com_amount, 2))

    tst_perf_global=[[] for i in range(n_clnt)]
    tst_perf_clnt=[[] for i in range(n_clnt)]
    
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        tst_perf_global = np.load('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnt = np.load('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, com_amount))

        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #test_perf_global berfore train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_global[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                ### train
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset, device)
                
                #test_perf_clnt after train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_clnt[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
            
            # Scale with weights
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0),device)

            
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset, device)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            
            if ((i+1) % save_period == 0):
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_all[i//save_period] = all_model
               
    return fed_mdls_all, tst_perf_all


def train_SCAFFOLD(data_obj, act_prob, learning_rate, batch_size, n_minibatch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, device, rand_seed=0, global_learning_rate=1):
    method_name = 'Scaffold'

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it
        
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_all = list(range(n_save_instances))
    
    tst_perf_all = np.zeros((com_amount, 2))
    tst_perf_global=[[] for i in range(n_clnt)]
    tst_perf_clnt=[[] for i in range(n_clnt)]

    n_par = len(get_mdl_params([model_func()])[0])
    state_param_list = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par    
    clnt_models = list(range(n_clnt))
    

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        tst_perf_global = np.load('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnt = np.load('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, com_amount))

        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        state_param_list = np.load('Output/%s/%s/%d_state_param_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):
            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([all_model], n_par)[0]

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #test_perf_global berfore train using local test
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_global[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                # Scale down c
                state_params_diff_curr = torch.tensor(-state_param_list[clnt] + state_param_list[-1]/weight_list[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_scaffold_mdl(clnt_models[clnt], model_func, state_params_diff_curr, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, n_minibatch, print_per, weight_decay, data_obj.dataset, device)
                #test_perf_clnt after train using local test
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_clnt[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                
                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_param_list[clnt] - state_param_list[-1] + 1/n_minibatch/learning_rate * (prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_param_list[clnt])*weight_list[clnt]
                state_param_list[clnt] = new_c
                clnt_params_list[clnt] = curr_model_param

            state_param_list[-1] += 1 / n_clnt * delta_c_sum

            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0),device)

            
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset, device)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            
            if ((i+1) % save_period == 0):
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)
                np.save('Output/%s/%s/%d_state_param_list.npy' %(data_obj.name, method_name, (i+1)), state_param_list)

                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period)):
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_state_param_list.npy' %(data_obj.name, method_name, i+1-save_period))

                        np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                        np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

            if ((i+1) % save_period == 0):
                fed_mdls_all[i//save_period] = all_model

    return fed_mdls_all, tst_perf_all

def train_FedDyn(data_obj, act_prob, learning_rate, batch_size, epoch, com_amount, print_per, weight_decay,  model_func, init_model, alpha_coef, save_period, lr_decay_per_round, device, rand_seed=0):
    
    method_name  = 'FedDyn' 
    
    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_all = list(range(n_save_instances)) # Avg all clients
    fed_mdls_cld = list(range(n_save_instances)) # Cloud models 
 
    tst_perf_all = np.zeros((com_amount, 2))

    tst_perf_global=[[] for i in range(n_clnt)]
    tst_perf_clnt=[[] for i in range(n_clnt)]

    n_par = len(get_mdl_params([model_func()])[0])
    
    local_param_list = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list  = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
        


    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    cld_model = model_func().to(device)
    cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    cld_mdl_param = get_mdl_params([cld_model], n_par)[0]
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_cld.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_cld[j] = fed_model
        
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        tst_perf_global = np.load('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnt = np.load('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, com_amount))

        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        local_param_list = np.load('Output/%s/%s/%d_local_param_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):
            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                # Train locally 
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                #test_perf_global berfore train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], model, data_obj.dataset, device)
                tst_perf_global[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                ### train
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt] # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_feddyn_mdl(model, model_func, alpha_coef_adpt, cld_mdl_param_tensor, local_param_list_curr, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset, device)
       
                #test_perf_clnt after train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_clnt[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                local_param_list[clnt] += curr_model_par-cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis = 0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

       
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0),device)
            cld_model = set_client_from_params(model_func().to(device), cld_mdl_param,device) 

           
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, cld_model, data_obj.dataset, device)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            rint("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            if ((i+1) % save_period == 0):
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                torch.save(cld_model.state_dict(), 'Output/%s/%s/%d_com_cld.pt' %(data_obj.name, method_name, (i+1)))

                np.save('Output/%s/%s/%d_local_param_list.npy' %(data_obj.name, method_name, (i+1)), local_param_list)
                np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

                if (i+1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                    os.remove('Output/%s/%s/%d_local_param_list.npy' %(data_obj.name, method_name, i+1-save_period))
                    os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

                    os.remove('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, i+1-save_period))
                    os.remove('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_all[i//save_period] = all_model
                fed_mdls_cld[i//save_period] = cld_model
            
    return fed_mdls_all, tst_perf_all, fed_mdls_cld

def train_FedProx(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, mu, lr_decay_per_round, device, rand_seed=0):
    method_name = 'FedProx'

    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
        
    # Average them based on number of datapoints (The one implemented)
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    fed_mdls_all = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    tst_perf_all = np.zeros((com_amount, 2))

    tst_perf_global=[[] for i in range(n_clnt)]
    tst_perf_clnt=[[] for i in range(n_clnt)]

    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
       
        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        tst_perf_global = np.load('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnt = np.load('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, com_amount))

        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
    
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            avg_model_param = get_mdl_params([avg_model], n_par)[0]
            avg_model_param_tensor = torch.tensor(avg_model_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #test_perf_global berfore train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_global[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                ### train
                clnt_models[clnt]= train_fedprox_mdl(clnt_models[clnt], avg_model_param_tensor, mu, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset, device)

                #test_perf_clnt after train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_clnt[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0),device)
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0),device)

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset, device)
           
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset, device)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
           
            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1)))     
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model

    return fed_mdls_all,  tst_perf_all

#train PFL
def train_pFedme(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, beta, lamda, K, mu, model_func, init_model, save_period, lr_decay_per_round, device, rand_seed=0):
    method_name = 'pFedMe'
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_all = list(range(n_save_instances))
    
    tst_perf_all = np.zeros((com_amount, 2))

    tst_perf_global=[[] for i in range(n_clnt)]
    tst_perf_clnt=[[] for i in range(n_clnt)]
    
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    local_params = list(range(n_clnt))
    personalized_params = list(range(n_clnt))
    

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        tst_perf_global = np.load('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnt = np.load('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, com_amount))

        
    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))
                if i==0:
                    local_params[clnt] = copy.deepcopy(list(clnt_models[clnt].parameters()))
                    personalized_params[clnt] = copy.deepcopy(list(clnt_models[clnt].parameters()))
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #test_perf_global berfore train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_global[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                ### train
                clnt_models[clnt], local_params[clnt], personalized_params[clnt]= train_pfedme_mdl(clnt_models[clnt],local_params[clnt], personalized_params[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, lamda, K, mu, data_obj.dataset, device)
                
                #test_perf_clnt after train using local test
                ###
                #replace client parameters by personalized params
                for param, personalized_param in zip (clnt_models[clnt].parameters(), personalized_params[clnt]):
                    param.data = personalized_param.data.clone() 
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_clnt[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))

                #replace client parameters by local params
                for param, local_param in zip (clnt_models[clnt].parameters(), local_params[clnt]):
                    param.data = local_param.data.clone() 
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
            
            # Scale with weights
            
            previous_global_model = copy.deepcopy(list(all_model.parameters()))
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0),device)
            #beta_aggregate_parameters
            for pre_param, param in zip(previous_global_model,all_model.parameters()):
                pre_param = pre_param.to('cpu')
                param = param.to('cpu')
                param.data = (1 - beta)*pre_param.data + beta*param.data 
            
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset, device)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            
            if ((i+1) % save_period == 0):
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_all[i//save_period] = all_model
               
    return fed_mdls_all, tst_perf_all

def train_ditto(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, plocal_steps, mu, model_func, init_model, save_period, lr_decay_per_round, device, rand_seed=0):
    method_name = 'ditto'
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_all = list(range(n_save_instances))
    
    tst_perf_all = np.zeros((com_amount, 2))

    tst_perf_global=[[] for i in range(n_clnt)]
    tst_perf_clnt=[[] for i in range(n_clnt)]
    
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    clnt_models_per = list(range(n_clnt))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        tst_perf_global = np.load('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnt = np.load('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, com_amount))

        
    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))
                if i==0:
                    clnt_models_per[clnt] = model_func().to(device)
                    clnt_models_per[clnt].load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                for params in clnt_models_per[clnt].parameters():
                    params.requires_grad = True
                #test_perf_global berfore train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_global[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                ### train
                clnt_models[clnt], clnt_models_per[clnt] = train_ditto_mdl(clnt_models[clnt], clnt_models_per[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, plocal_steps, mu, data_obj.dataset, device)
                
                #test_perf_clnt after train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models_per[clnt], data_obj.dataset, device)
                tst_perf_clnt[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
            
            # Scale with weights
            
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0),device)
            
           
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset, device)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            
            if ((i+1) % save_period == 0):
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_all[i//save_period] = all_model
               
    return fed_mdls_all, tst_perf_all

def train_ala(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, eta, rand_percent, layer_idx,  model_func, init_model, save_period, lr_decay_per_round, device, rand_seed=0):
    method_name = 'ala'
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_all = list(range(n_save_instances))
    
    tst_perf_all = np.zeros((com_amount, 2))

    tst_perf_global=[[] for i in range(n_clnt)]
    tst_perf_clnt=[[] for i in range(n_clnt)]
    
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        tst_perf_global = np.load('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnt = np.load('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, com_amount))

        
    else:
        clnt_alas = dict()
        for i in range(0, n_clnt):
            clnt_alas[i] = None

        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                if i==0:
                    clnt_models[clnt] = model_func().to(device)
                    clnt_models[clnt].load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #test_perf_global berfore train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_global[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                ### train
                clnt_models[clnt] = train_ala_mdl(clnt_models[clnt],copy.deepcopy(all_model), trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, eta, rand_percent, layer_idx, clnt_alas, clnt ,data_obj.dataset, device)
                
                #test_perf_clnt after train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_clnt[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
            
            # Scale with weights
            
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0),device)
            
            
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset, device)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            
            if ((i+1) % save_period == 0):
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                # np.save('Output/%s/%s/%d_com_trn_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), trn_perf_clnt)
                np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_all[i//save_period] = all_model
               
    return fed_mdls_all, tst_perf_all

def train_fedrod(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, num_classes, model_func, init_model, save_period, lr_decay_per_round, device, rand_seed=0):
    method_name = 'fedrod'
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_all = list(range(n_save_instances))
    
    tst_perf_all = np.zeros((com_amount, 2))

    tst_perf_global=[[] for i in range(n_clnt)]
    tst_perf_clnt=[[] for i in range(n_clnt)]
    
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    clnt_head = list(range(n_clnt))
    clnt_head_per = list(range(n_clnt))

    all_model = model_func().to(device)
    all_head = copy.deepcopy(all_model.fc)
    all_model.fc = nn.Identity()
    all_model = BaseHeadSplit(all_model,all_head)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            
            fed_model = model_func()
            
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        tst_perf_global = np.load('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnt = np.load('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, com_amount))

        
    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                
                clnt_models[clnt] = model_func().to(device)
                clnt_head[clnt] = copy.deepcopy(clnt_models[clnt].fc)
                clnt_models[clnt].fc = nn.Identity()
                clnt_models[clnt] = BaseHeadSplit(clnt_models[clnt],clnt_head[clnt])
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))
                if i==0:
                    clnt_head_per[clnt]=copy.deepcopy(clnt_head[clnt])
                    
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #test_perf_global berfore train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_global[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                ### train
                clnt_models[clnt], clnt_head_per[clnt] = train_fedrod_mdl(clnt_models[clnt],clnt_head_per[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, num_classes, data_obj.dataset, device)
              
                #test_perf_clnt after train using local test
                loss_tst, acc_tst = get_acc_loss_fedrod(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt],clnt_head_per[clnt], data_obj.dataset, device)
                tst_perf_clnt[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
            
            # Scale with weights
            mf=model_func()
            mf_head = copy.deepcopy(mf.fc)
            mf.fc = nn.Identity()
            mf = BaseHeadSplit(mf, mf_head)
            all_model = set_client_from_params(mf, np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0),device)
            
           
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset, device)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            
            if ((i+1) % save_period == 0):
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        
                        os.remove('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_all[i//save_period] = all_model
               
    return fed_mdls_all, tst_perf_all

def train_fedrecon(data_obj, act_prob ,learning_rate, batch_size,
                   recon_epochs, pers_epochs, recon_lr, pers_lr, s_lr, n_test_client, supset_ratio, 
                   com_amount, print_per, model_func, init_model, save_period, lr_decay_per_round, device, rand_seed=0):
    method_name = 'fedrecon'
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_all = list(range(n_save_instances))
    
    tst_perf_all = np.zeros((com_amount, 2))

    tst_perf_global=[[] for i in range(n_clnt)]
    tst_perf_clnt=[[] for i in range(n_clnt)]
    
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_diff_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    clnt_head = list(range(n_clnt))
    

    all_model = model_func().to(device)
    all_head = copy.deepcopy(all_model.fc)
    all_model.fc = nn.Identity()
    all_model = BaseHeadSplit(all_model,all_head)
    
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            
            
            fed_model = model_func()
            
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        tst_perf_global = np.load('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnt = np.load('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, com_amount))

        
    else:
        for i in range(com_amount):

            inc_seed = 0

            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                if i==0:
                    clnt_models[clnt] = model_func().to(device)
                    clnt_head[clnt] = copy.deepcopy(clnt_models[clnt].fc)
                    clnt_models[clnt].fc = nn.Identity()
                    clnt_models[clnt] = BaseHeadSplit(clnt_models[clnt],clnt_head[clnt])
                    clnt_models[clnt].head = copy.deepcopy(all_model.head)
                clnt_models[clnt].base = copy.deepcopy(all_model.base)
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #test_perf_global berfore train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_global[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                ### train
                clnt_models[clnt] = train_fedrecon_mdl(clnt_models[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size,
                                                       recon_epochs, pers_epochs, recon_lr,pers_lr, supset_ratio,
                                                       print_per, data_obj.dataset, device)
                
                #test_perf_clnt after train using local test
                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.clnt_x_test[clnt], data_obj.clnt_y_test[clnt], clnt_models[clnt], data_obj.dataset, device)
                tst_perf_clnt[clnt].append([loss_tst, acc_tst])
                print("**** Communication sel %3d, client %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, clnt, acc_tst, loss_tst))
                
                clnt_diff_params_list[clnt] = get_diff_mdl_params([clnt_models[clnt].base], all_model.base, n_par)[0]
                

            
            # Scale with weights
            mf=model_func()
            mf_head = copy.deepcopy(mf.fc)
            mf.fc = nn.Identity()
            mf = BaseHeadSplit(mf, mf_head)
            all_model.base = set_client_from_diff_params(mf.base, all_model.base, np.sum(clnt_diff_params_list*weight_list/np.sum(weight_list), axis = 0),s_lr,device)
            
            
            loss_tst, acc_tst = get_acc_loss_fedrecon(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset, device, n_test_client, recon_epochs, supset_ratio ,recon_lr, batch_size)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
          
            if ((i+1) % save_period == 0):
                # torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1))) 
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                np.save('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, (i+1)), tst_perf_global)
                # np.save('Output/%s/%s/%d_com_trn_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), trn_perf_clnt)
                np.save('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, (i+1)), tst_perf_clnt)

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_com_tst_perf_global.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnt.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_all[i//save_period] = all_model
               
    return fed_mdls_all, tst_perf_all
