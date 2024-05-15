from gym.envs.my_env.FL.utils.utils_libs import *
from gym.envs.my_env.FL.utils.utils_datasets import *
from gym.envs.my_env.FL.utils.utils_models import *
from gym.envs.my_env.FL.utils.utils_general import *

### Methods
def train_Fed(self, action, s_action, com_count):
    device=self.device
    method_name = 'Fed'
    n_clnt=self.data_obj.n_client
    client_action=action
    server_action=np.asarray(s_action[0])
    
    weight_list = np.asarray([len(self.data_obj.clnt_train_loader[i])*self.batch_size for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    server_action=server_action.reshape((n_clnt,1))
    state_client=[]
    state_server=[]
    reward_client=[]
    reward_server=[]

    if not os.path.exists('%s/%s/%s' %(self.path, self.data_obj.name, method_name)):
        os.mkdir('%s/%s/%s' %(self.path, self.data_obj.name, method_name))
        
    n_par = len(get_mdl_params([self.model_func()])[0])

    init_par_list=get_mdl_params([self.init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    
    
    all_model = self.model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(self.init_model.named_parameters())))
    
    if os.path.exists('%s/%s/%s/%d_com_tst_perf_all.npy' %(self.path, self.data_obj.name, method_name, self.com_amount)):
        # Load performances and models...
        for j in range(self.n_save_instances):
            
            fed_model = self.model_func()
            fed_model.load_state_dict(torch.load('%s/%s/%s/%d_com_all.pt' %(self.path, self.data_obj.name, method_name, (j+1)*self.save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            self.fed_mdls_all[j] = fed_model
        
        self.tst_perf_all = np.load('%s/%s/%s/%d_com_tst_perf_all.npy' %(self.path, self.data_obj.name, method_name, self.com_amount))
        
        clnt_params_list = np.load('%s/%s/%s/%d_clnt_params_list.npy' %(self.path, self.data_obj.name, method_name, self.com_amount))
        
    else:
        i=com_count
        inc_seed = 0
        while(True):
            # Fix randomness in client selection
            np.random.seed(i + self.rand_seed + inc_seed)
            act_list    = np.random.uniform(size=n_clnt)
            act_clients = act_list <= self.act_prob
            selected_clnts = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clnts) != 0:
                break
        print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

        for clnt in selected_clnts:
            print('---- Training client %d' %clnt)
            self.learning_rate=client_action[clnt][0]
            self.epoch=int(client_action[clnt][-1])
            clnt_models[clnt] = self.model_func().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))

            for params in clnt_models[clnt].parameters():
                params.requires_grad = True
            loss_tst, acc_tst, acc5_tst = get_acc_loss_(self.data_obj.clnt_test_loader[clnt], clnt_models[clnt], self.data_obj.dataset,device)
            loss_tst+=self.penalty
            state_client.append([acc_tst])
            self.tst_perf_global[clnt].append([loss_tst, acc_tst])
            print("**** Communication global client test %3d, client %3d, Test Accuracy: %.4f Acc5:%.4f, Loss: %.4f" %(i+1, clnt, acc_tst, acc5_tst, loss_tst))
            
            clnt_models[clnt],acc_trn,loss_trn = train_model_(clnt_models[clnt], self.data_obj.clnt_train_loader[clnt], self.learning_rate * (self.lr_decay_per_round ** i), self.batch_size, self.epoch, self.print_per, self.weight_decay, self.data_obj.dataset, device)
            self.trn_perf_clnt[clnt].append([loss_trn,acc_trn])
            
            loss_tst, acc_tst, acc5_tst = get_acc_loss_(self.data_obj.clnt_test_loader[clnt], clnt_models[clnt], self.data_obj.dataset, device)
            loss_tst+=self.penalty
            reward_client.append(1/loss_tst)
            self.tst_perf_clnt[clnt].append([loss_tst, acc_tst])
            print("**** Communication client test %3d, client %3d, Test Accuracy: %.4f Acc5:%.4f, Loss: %.4f" %(i+1, clnt, acc_tst, acc5_tst, loss_tst))
            clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            loss_tst, acc_tst, acc5_tst = get_acc_loss_(self.data_obj.test_loader, clnt_models[clnt], self.data_obj.dataset, device)
            loss_tst+=self.penalty
            state_server.append(acc_tst)
            self.tst_perf_clnt_[clnt].append([loss_tst, acc_tst])
            print("**** Communication client global test %3d, client %3d, Test Accuracy: %.4f Acc5:%.4f, Loss: %.4f" %(i+1, clnt, acc_tst, acc5_tst, loss_tst))
        aggregation_weight=s_action[0]
        # Scale with weights
        print("weight_list:")
        print(weight_list)
        print("weight_list_sum:")
        print(np.sum(weight_list))
        print("server_action:")
        print(server_action)
        print("server_action_sum:")
        print(np.sum(server_action))
        weight_list=server_action
        all_model = set_client_from_params(self.model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0),device)
        self.init_model=all_model
        
        loss_tst, acc_tst, acc5_tst = get_acc_loss_(self.data_obj.test_loader, all_model, self.data_obj.dataset,device)
        loss_tst+=self.penalty
        reward_server.append(1/loss_tst)
        self.tst_perf_all[i] = [loss_tst, acc_tst]
        print("**** Communication all %3d, Test Accuracy: %.4f Acc5:%.4f, Loss: %.4f" %(i+1, acc_tst, acc5_tst, loss_tst))
        
        if ((i+1) % self.save_period == 0):
            torch.save(all_model.state_dict(), '%s/%s/%s/%d_com_all.pt' %(self.path, self.data_obj.name, method_name, (i+1)))
            np.save('%s/%s/%s/%d_clnt_params_list.npy' %(self.path, self.data_obj.name, method_name, (i+1)), clnt_params_list)

            np.save('%s/%s/%s/%d_com_tst_perf_global.npy' %(self.path, self.data_obj.name, method_name, (i+1)), self.tst_perf_global)
            np.save('%s/%s/%s/%d_com_trn_perf_clnt.npy' %(self.path, self.data_obj.name, method_name, (i+1)), self.trn_perf_clnt)
            np.save('%s/%s/%s/%d_com_tst_perf_clnt.npy' %(self.path, self.data_obj.name, method_name, (i+1)), self.tst_perf_clnt)
            np.save('%s/%s/%s/%d_com_tst_perf_clnt_.npy' %(self.path, self.data_obj.name, method_name, (i+1)), self.tst_perf_clnt_)
            np.save('%s/%s/%s/%d_com_tst_perf_all.npy' %(self.path, self.data_obj.name, method_name, (i+1)), self.tst_perf_all[:i+1])

           
            if (i+1) > self.save_period:
                if os.path.exists('%s/%s/%s/%d_com_trn_perf_all.npy' %(self.path, self.data_obj.name, method_name, i+1-self.save_period)):
                    # Delete the previous saved arrays
                    os.remove('%s/%s/%s/%d_com_tst_perf_global.npy' %(self.path, self.data_obj.name, method_name, i+1-self.save_period))
                    os.remove('%s/%s/%s/%d_com_trn_perf_clnt.npy' %(self.path, self.data_obj.name, method_name, i+1-self.save_period))
                    os.remove('%s/%s/%s/%d_tst_perf_clnt.npy' %(self.path, self.data_obj.name, method_name, i+1-self.save_period))
                    os.remove('%s/%s/%s/%d_com_tst_perf_clnt_.npy' %(self.path, self.data_obj.name, method_name, i+1-self.save_period))
                    os.remove('%s/%s/%s/%d_com_tst_perf_all.npy' %(self.path, self.data_obj.name, method_name, i+1-self.save_period))

                    os.remove('%s/%s/%s/%d_clnt_params_list.npy' %(self.path, self.data_obj.name, method_name, i+1-self.save_period))

            if ((i+1) % self.save_period == 0):
                self.fed_mdls_all[i//self.save_period] = all_model
    state_server=[state_server]            
    return state_client,state_server,reward_client, reward_server

def reset_Fed(self):
    device = self.device 
    method_name = 'Fed'
    n_clnt=self.data_obj.n_client
    
    weight_list = np.asarray([len(self.data_obj.clnt_train_loader[i])*self.batch_size for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    state_client=[]
    state_server=[]
    reward_client=[]
    reward_server=[]

    if not os.path.exists('%s/%s/%s' %(self.path, self.data_obj.name, method_name)):
        os.makedirs('%s/%s/%s' %(self.path, self.data_obj.name, method_name))
        
    n_par = len(get_mdl_params([self.model_func()])[0])

    init_par_list=get_mdl_params([self.init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    
    
    all_model = self.model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(self.init_model.named_parameters())))
    
    if os.path.exists('%s/%s/%s/%d_com_tst_perf_all.npy' %(self.path, self.data_obj.name, method_name, self.com_amount)):
        # Load performances and models...
        for j in range(self.n_save_instances):
            
            fed_model = self.model_func()
            fed_model.load_state_dict(torch.load('%s/%s/%s/%d_com_all.pt' %(self.path, self.data_obj.name, method_name, (j+1)*self.save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            self.fed_mdls_all[j] = fed_model
        
        self.tst_perf_all = np.load('%s/%s/%s/%d_com_tst_perf_all.npy' %(self.path, self.data_obj.name, method_name, self.com_amount))
        
        clnt_params_list = np.load('%s/%s/%s/%d_clnt_params_list.npy' %(self.path, self.data_obj.name, method_name, self.com_amount))
        
    else:
        i=2023
        inc_seed = 0
        while(True):
            np.random.seed(i + self.rand_seed + inc_seed)
            act_list    = np.random.uniform(size=n_clnt)
            act_clients = act_list <= self.act_prob
            selected_clnts = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clnts) != 0:
                break
        print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

        for clnt in selected_clnts:
            print('---- Training client %d' %clnt)
            clnt_models[clnt] = self.model_func().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))

            for params in clnt_models[clnt].parameters():
                params.requires_grad = True
            loss_tst, acc_tst, acc5_tst = get_acc_loss_(self.data_obj.clnt_test_loader[clnt], clnt_models[clnt], self.data_obj.dataset,device)
            loss_tst+=self.penalty
            state_client.append(acc_tst)
            print("**** Communication global client test %3d, client %3d, Test Accuracy: %.4f Acc5:%.4f, Loss: %.4f" %(i+1, clnt, acc_tst, acc5_tst, loss_tst))
            
            clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            loss_tst, acc_tst, acc5_tst = get_acc_loss_(self.data_obj.test_loader, clnt_models[clnt], self.data_obj.dataset,device)
            loss_tst+=self.penalty
            state_server.append(acc_tst)
            print("**** Communication client global test %3d, client %3d, Test Accuracy: %.4f Acc5:%.4f, Loss: %.4f" %(i+1, clnt, acc_tst, acc5_tst, loss_tst))
    state_client.extend(state_server)
    state=state_client
    return state
