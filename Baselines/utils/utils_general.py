from .utils_libs import *
from .utils_dataset import *
from .utils_models import *
from .utils_optimizer import *
from .utils_ALA import ALA
from .utils_ALA import ALA_
max_norm = 10
# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name,device, w_decay = None):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()            
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
    
    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)
        
    model.train()
    return loss_overall, acc_overall / n_tst
def get_acc_loss_fedrecon(data_x, data_y, model, dataset_name, device, n_test_client,epoch, supset_ratio ,recon_lr, batch_size):
    client_data_x = np.array_split(data_x,n_test_client)
    client_data_y = np.array_split(data_y,n_test_client)
    total_acc=0.
    total_loss=0.
    for i in range(n_test_client):
        loss_fn = nn.CrossEntropyLoss()
        client_model = copy.deepcopy(model)
        opt_head = torch.optim.SGD(client_model.head.parameters(),lr=recon_lr)
        num_rec = int(len(client_data_x[i])*0.7* supset_ratio)
        client_data_x_rec = client_data_x[i][0:num_rec]
        client_data_y_rec = client_data_y[i][0:num_rec]
        client_data_x_tst = client_data_x[i][num_rec:]
        client_data_y_tst = client_data_y[i][num_rec:]
        recon_trn_gen = data.DataLoader(Dataset(client_data_x_rec,client_data_y_rec,train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
        
        client_model.train(); model= model.to(device)
        # reconstruction
        for e in range(epoch):
            #Training
            recon_trn_gen_iter = recon_trn_gen.__iter__()
            for i in range(int(np.ceil(num_rec/batch_size))):
                batch_x,batch_y = recon_trn_gen_iter.__next__()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                rep = client_model.base(batch_x)
                out_g = client_model.head(rep)
                loss_recon = loss_fn(out_g, batch_y.reshape(-1).long())
                opt_head.zero_grad()
                loss_recon.backward()
                opt_head.step()
        # test       
        acc_overall = 0; loss_overall = 0;
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        n_tst = client_data_x_tst.shape[0]
        batch_size = min(6000, n_tst)
        
        tst_gen = data.DataLoader(Dataset(client_data_x_tst, client_data_y_tst, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
        client_model.eval(); client_model = client_model.to(device)
        with torch.no_grad():
            tst_gen_iter = tst_gen.__iter__()
            for i in range(int(np.ceil(n_tst/batch_size))):
                batch_x, batch_y = tst_gen_iter.__next__()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                y_pred = client_model(batch_x)
                
                loss = loss_fn(y_pred, batch_y.reshape(-1).long())
                loss_overall += loss.item()
                # Accuracy calculation
                y_pred = y_pred.cpu().numpy()            
                y_pred = np.argmax(y_pred, axis=1).reshape(-1)
                batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
                batch_correct = np.sum(y_pred == batch_y)
                acc_overall += batch_correct
        loss_overall /= n_tst
        acc_overall /= n_tst
        total_loss+=loss_overall
        total_acc+= acc_overall    
        client_model.train()
    return total_loss/n_test_client,  total_acc/n_test_client
def get_acc_loss_fedrod(data_x, data_y, model,head, dataset_name,device):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            rep = model.base(batch_x)
            out_g = model.head(rep)
            out_p = head(rep.detach())
            y_pred = out_g.detach()+out_p
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()            
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
    
    loss_overall /= n_tst
        
    model.train()
    return loss_overall, acc_overall / n_tst
# --- Helper functions

def set_client_from_params(mdl, params, device):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length
    
    mdl.load_state_dict(dict_param)    
    return mdl
def set_client_from_diff_params(mdl, params,s_lr, device):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.add_(s_lr*torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length
    
    mdl.load_state_dict(dict_param)    
    return mdl

def get_mdl_params(model_list, n_par=None):
    
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def diff_clnt_model_with_global_model(client_model: nn.modules, global_model: nn.modules):

    with torch.no_grad():
        for ((_ , param_local), (_, param_global)) in zip(client_model.named_parameters(), global_model.named_parameters()):
            param_local.sub_(param_global)


def aggregation_from_clnt_model_diffs(global_model, diff_models, weight_list, s_lr):

    with torch.no_grad():
        weight_sum = sum(weight_list)
        weight_list = list(map(lambda w: w / weight_sum, weight_list))
        for model, weight in zip(diff_models, weight_list):
            for _, param in model.named_parameters():
                param.data = weight[0] * param.data # torch.from_numpy

        # aggregate model params
        weighted_diff = []
        for model in diff_models:
            for ((_, cln_param), (_, global_param)) in zip(model.named_parameters(), global_model.named_parameters()):
                global_param.sub_(s_lr * cln_param.data)

def get_diff_mdl_params(model_list, global_model,n_par=None):
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for (name, param), (g_name, g_param) in zip(mdl.named_parameters(), global_model.named_parameters()):
            temp = param.data.cpu().numpy().reshape(-1).sub_(g_param.data.cpu().numpy().reshape(-1)) 
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)
# --- Train functions

def train_model(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, device):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    for e in range(epoch):
        # Training
        
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

        if (e+1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" %(e+1, acc_trn, loss_trn))
            model.train()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_scaffold_mdl(model, model_func, state_params_diff, trn_x, trn_y, learning_rate, batch_size, n_minibatch, print_per, weight_decay, dataset_name, device):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    n_iter_per_epoch = int(np.ceil(n_trn/batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)
    count_step = 0
    is_done = False
    
    step_loss = 0; n_data_step = 0
    for e in range(epoch):
        # Training
        if is_done:
            break
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = torch.sum(local_par_list * state_params_diff)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0]; n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay)/2 * np.sum(params * params)
                print("Step %3d, Training Loss: %.4f" %(count_step, step_loss))
                step_loss = 0; n_data_step = 0
                model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_feddyn_mdl(model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, device):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef+weight_decay)/2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

###
def train_fedprox_mdl(model, avg_model_param_, mu, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, device):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = len(avg_model_param_)
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = mu/2 * torch.sum(local_par_list * local_par_list)
            loss_algo += -mu * torch.sum(local_par_list * avg_model_param_)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += weight_decay/2 * np.sum(params * params)
            
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def accuracy(output, target, device, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        # batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
# --- Evaluate a NN model
def get_acc_loss_(test_loader, model, dataset_name, device, w_decay = None):
    acc1_overall = 0; acc5_overall=0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    n_tst = 0
    model.eval(); model = model.to(device)
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            # compute output
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = loss_fn(output, target)
            loss_overall+=loss.item()
            n_tst+=target.size(0)
            #Accuracy calculation
            acc1, acc5 = accuracy(output, target,device, topk=(1,5))
            acc1_overall+=acc1.item()
            acc5_overall+=acc5.item()
    
    loss_overall /= n_tst
    acc1_overall /= n_tst
    acc5_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)
        
    model.train()
    return loss_overall, acc1_overall, acc5_overall
def get_acc_loss_fedrecon_(test_loader, model, dataset_name, device, n_test_client,epoch, supset_ratio, recon_lr, batch_size):
    loss_overall = 0.
    acc1_overall = 0.
    acc5_overall = 0.
    data_indices = [i for i in range(0, len(test_loader))]
    clients_data_indices = np.array_split(data_indices, n_test_client)
    for i in range(n_test_client):
        loss_overall_ = 0.
        acc1_overall_ = 0.
        acc5_overall_ = 0.
        n_tst = 0
        loss_fn = nn.CrossEntropyLoss()
        client_model = copy.deepcopy(model)
        opt_head = torch.optim.SGD(client_model.head.parameters(),lr=recon_lr)
        num_rec = int(len(clients_data_indices[i])*0.7*supset_ratio)
        client_data_rec_range = clients_data_indices[i][0:num_rec]
        
        client_model.train(); model= model.to(device)
        # reconstruction
        for e in range(epoch):
            #Training
            for i, (images, target) in enumerate(test_loader):
                if i not in client_data_rec_range: continue
                # compute outpu
                images = images.to(device)
                target = target.to(device)

                rep = client_model.base(images)
                out_g = client_model.head(rep)
                loss_recon = loss_fn(out_g, target)
                opt_head.zero_grad()
                loss_recon.backward()
                opt_head.step()
        # test       
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        
        client_model.eval(); client_model = client_model.to(device)
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
                if i in client_data_rec_range: continue
                images = images.to(device)
                target = target.to(device)
                output = client_model(images)
                loss = loss_fn(output, target)
                loss_overall_ += loss.item()
                n_tst += target.size(0)
                # Accuracy calculation
                acc1, acc5 = accuracy(output, target, device, topk=(1,5))
                acc1_overall_ += acc1.item()
                acc5_overall_ += acc5.item()
        loss_overall_ /= n_tst
        acc1_overall_ /= n_tst
        acc5_overall_ /= n_tst
        loss_overall += loss_overall_
        acc1_overall += acc1_overall_
        acc5_overall += acc5_overall_
        client_model.train()

    loss_overall /= n_test_client
    acc1_overall /= n_test_client
    acc5_overall /= n_test_client

    return loss_overall, acc1_overall, acc5_overall

def get_acc_loss_fedrod_(test_loader, model,head, dataset_name, device):
    acc1_overall = 0; acc5_overall=0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    n_tst = 0
    model.eval(); model = model.to(device)
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            # compute output
            images = images.to(device)
            target = target.to(device)
            rep = model.base(images)
            out_g = model.head(rep)
            out_p = head(rep.detach())
            output = out_g.detach()+out_p
            
            loss = loss_fn(output, target)
            loss_overall+=loss.item()
            n_tst+=target.size(0)
            #Accuracy calculation
            acc1, acc5 = accuracy(output, target,device, topk=(1,5))
            acc1_overall+=acc1.item()
            acc5_overall+=acc5.item()
    
    loss_overall /= n_tst
    acc1_overall /= n_tst
    acc5_overall /= n_tst
         
    model.train()
    return loss_overall, acc1_overall, acc5_overall
# ---Train_imagenet functions
def train_model_(model, train_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, device):
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    for e in range(epoch):
        # Training
        for i, (images, target) in enumerate(train_loader):
            # compute output
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = loss_fn(output, target)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            
        if (e+1) % print_per == 0:
            loss_trn, acc1_trn, acc5_trn = get_acc_loss_(train_loader, model, weight_decay, device)
            print("Epoch %3d, Training Acc1: %.4f Acc5:%.4f, Loss: %.4f" %(e+1, acc1_trn, acc5_trn, loss_trn))
            model.train()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_scaffold_mdl_(model, model_func, state_params_diff, train_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, device):
   
    n_minibatch = (epoch * len(train_loader))
    print_per = print_per * len(train_loader)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    count_step = 0
    is_done = False
    
    step_loss = 0; n_data_step = 0
    for e in range(epoch):
        # Training
        if is_done:
            break
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            output = model(images)
            # Get f_i estimate
            loss_f_i = loss_fn(output,target)
            
            #Get linear penalty on the current parameters estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = torch.sum(local_par_list * state_params_diff)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            step_loss += loss.item() * list(target.size())[0]; n_data_step += list(target.size())[0]

        if (count_step) % print_per == 0:
            step_loss /= n_data_step
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                step_loss += (weight_decay)/2 * np.sum(params * params)
            print("Step %3d, Training Loss: %.4f" %(count_step, step_loss))
            step_loss = 0; n_data_step = 0
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()          
    return model

def train_feddyn_mdl_(model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, train_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, device):
    n_trn = len(train_loader)*batch_size
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    for e in range(epoch):
        #Training
        epoch_loss=0.
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)
            #compute output
            output = model(images)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(output, target)
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(target.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef+weight_decay)/2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
        
    return model

###
def train_fedprox_mdl_(model, avg_model_param_, mu, train_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, device):
    n_trn = len(train_loader)*batch_size
    # trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = len(avg_model_param_)
    
    for e in range(epoch):
        #Traininig 
        epoch_loss=0.
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)
            #compute output
            output = model(images)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(output, target)
                    
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = mu/2 * torch.sum(local_par_list * local_par_list)
            loss_algo += -mu * torch.sum(local_par_list * avg_model_param_)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(target.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += weight_decay/2 * np.sum(params * params)
            
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model


#PFL train
# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

def train_pfedme_mdl(model, local_params, personalized_params, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, lamda, K, mu, dataset_name, device):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x,trn_y,train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    #these parameters are for personalized federated learning.
    
    optimizer = pFedMeOptimizer(
        model.parameters(), lr=learning_rate, lamda = lamda, mu = mu
    )
    model.train(); model = model.to(device)
    
    for e in range(epoch):
        # Training 
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x,batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # K is number of personalized steps
            for j in range(K): 
                y_pred=model(batch_x)
                loss = loss_fn(y_pred,batch_y.reshape(-1).long())
                loss = loss / list(batch_y.size())[0]
                optimizer.zero_grad()
                loss.backward()
                # finding aproximate theta
                personalized_params = optimizer.step(local_params,device)
            
            #update local weight after fing aproximate theta
            for new_param, localweight in zip(personalized_params, local_params):
                localweight = localweight.to(device)
                localweight.data = localweight.data - lamda * learning_rate *(localweight.data - new_param.data)
    
    #update parameters
    for param, new_param in zip (model.parameters(), local_params):
        param.data = new_param.data.clone() 
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    
    return model, local_params, personalized_params
                
def train_pfedme_mdl_(model,local_params, personalized_params, train_loader, learning_rate, batch_size, epoch, print_per, lamda, K, mu, dataset_name, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    
    #these parameters are for personalized federated learning.
    
    optimizer = pFedMeOptimizer(
        model.parameters(), lr=learning_rate, lamda = lamda, mu = mu
    )
    model.train(); model = model.to(device)
    
    for e in range(epoch):
        # Training
        for i, (images, target) in enumerate(train_loader):
            # compute output
            images = images.to(device)
            target = target.to(device)
            # K is number of personalized steps
            for j in range(K):
                
                output = model(images)
                loss = loss_fn(output, target)
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
                personalized_params = optimizer.step(local_params,device)
            
            #update local weight after fing aproximate theta
            for new_param, localweight in zip(personalized_params, local_params):
                localweight = localweight.to(device)
                localweight.data = localweight.data - lamda * learning_rate *(localweight.data - new_param.data)
    #update parameters
    for param, new_param in zip (model.parameters(), local_params):
        param.data = new_param.data.clone() 
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model,local_params, personalized_params

def train_ditto_mdl(model, model_per, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, plocal_steps, mu, dataset_name, device):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x,trn_y,train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    
    optimizer_per = PerturbedGradientDescent(
            model_per.parameters(), lr=learning_rate, mu=mu)
    model_per.train(); model_per = model_per.to(device)
    model.train(); model = model.to(device)
    # p_train
    for e in range(plocal_steps):
        #Training
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x,batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred=model_per(batch_x)
            loss = loss_fn(y_pred,batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            optimizer_per.zero_grad()
            loss.backward()
            optimizer_per.step(model.parameters(),device)
    # train
    for e in range(epoch):
        # Training 
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x,batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred=model(batch_x)
            loss = loss_fn(y_pred,batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #Freeze model
    for params in model_per.parameters():
        params.requires_grad = False
    model_per.eval()
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model,model_per

def train_ditto_mdl_(model, model_per, train_loader, learning_rate, batch_size, epoch, print_per, plocal_steps, mu, dataset_name, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    
    optimizer_per = PerturbedGradientDescent(
            model_per.parameters(), lr=learning_rate, mu=mu)
    model_per.train(); model_per = model_per.to(device)
    model.train(); model = model.to(device)
    # p_train
    for e in range(plocal_steps):
        #Training
        for i, (images, target) in enumerate(train_loader):
            # compute output
            images = images.to(device)
            target = target.to(device)
            
            output = model_per(images)
            loss = loss_fn(output, target)
            optimizer_per.zero_grad()
            loss.backward()
            optimizer_per.step(model.parameters(),device)
    # train
    for e in range(epoch):
        # Training 
        for i, (images, target) in enumerate(train_loader):
            # compute output
            images = images.to(device)
            target = target.to(device)
            
            output = model(images)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #Freeze model
    for params in model_per.parameters():
        params.requires_grad = False
    model_per.eval()
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model, model_per

def train_ala_mdl(model, global_model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, eta, rand_percent, layer_idx, clnt_alas , clnt_id,  dataset_name, device):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x,trn_y,train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    if clnt_alas[clnt_id] == None:
        ala = ALA(loss_fn, trn_x, trn_y, dataset_name, batch_size, rand_percent, layer_idx, eta, device) 
        clnt_alas[clnt_id] = ala
    else:
        ala = clnt_alas[clnt_id]

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    model.train(); model = model.to(device)
    
    # local_initialization
    ala.adaptive_local_aggregation(global_model, model)
    
    for e in range(epoch):
        # Training 
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x,batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred=model(batch_x)
            loss = loss_fn(y_pred,batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    #Freese model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    
    return model

def train_ala_mdl_(model, global_model, train_loader, learning_rate, batch_size, epoch, print_per, eta, rand_percent, layer_idx,  clnt_alas, clnt_id,  dataset_name, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    # local_initialization

    if clnt_alas[clnt_id] == None:
        ala = ALA_(loss_fn, train_loader, batch_size, rand_percent, layer_idx, eta, device) 
        clnt_alas[clnt_id] = ala
    else:
        ala = clnt_alas[clnt_id]


    ala.adaptive_local_aggregation(global_model, model)
      
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    model.train(); model = model.to(device)
       
    for e in range(epoch):
        # Training 
        for i, (images, target) in enumerate(train_loader):
            # compute output
            images = images.to(device)
            target = target.to(device)
            
            output = model(images)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            
    #Freese model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    
    return model

# head as personal models' head
def train_fedrod_mdl(model, head, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, num_classes, dataset_name, device):
    opt_head = torch.optim.SGD(head.parameters(),lr=learning_rate)
    sample_per_class = torch.zeros(num_classes)
    n_trn = trn_x.shape[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    trn_gen = data.DataLoader(Dataset(trn_x,trn_y,train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    trn_gen_iter = trn_gen.__iter__()
    for i in range(int(np.ceil(n_trn/batch_size))):
        batch_x, batch_y = trn_gen_iter.__next__()
        for b_y in batch_y.reshape(-1).long().numpy().tolist():
            sample_per_class[b_y] += 1
    
    model.train(); model = model.to(device)
    
    
    for e in range(epoch):
        # Training 
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x,batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            rep = model.base(batch_x)
            out_g = model.head(rep)
            loss_bsm = balanced_softmax_loss(batch_y.reshape(-1).long(), out_g, sample_per_class)
            
            optimizer.zero_grad()
            loss_bsm.backward()
            optimizer.step()
            
            out_p = head(rep.detach())
            loss = loss_fn(out_g.detach()+out_p, batch_y.reshape(-1).long())
            opt_head.zero_grad()
            loss.backward()
            opt_head.step()
    #Freese model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    
    return model, head

def train_fedrod_mdl_(model, head, train_loader, learning_rate, batch_size, epoch, print_per, num_classes, dataset_name, device):

    opt_head = torch.optim.SGD(head.parameters(),lr=learning_rate)
    sample_per_class = torch.zeros(num_classes)
    # n_trn = trn_x.shape[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for i, (images, target) in enumerate(train_loader):
        for t in target:
            sample_per_class[t] += 1
    model.train(); model = model.to(device)
    
    
    for e in range(epoch):
        # Training 
        for i, (images, target) in enumerate(train_loader):
            # compute output
            images = images.to(device)
            target = target.to(device)
            
            rep = model.base(images)
            out_g = model.head(rep)
            loss_bsm = balanced_softmax_loss(target, out_g, sample_per_class)
            
            optimizer.zero_grad()
            loss_bsm.backward()
            optimizer.step()
            
            out_p = head(rep.detach())
            loss = loss_fn(out_g.detach()+out_p, target)
            opt_head.zero_grad()
            loss.backward()
            opt_head.step()
    
    #Freese model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    
    return model, head



def train_fedrecon_mdl(model, trn_x, trn_y, learning_rate, batch_size,
                       recon_epochs, pers_epochs, recon_lr, pers_lr, supset_ratio,
                       print_per, dataset_name, device):
    
    opt_head = torch.optim.SGD(model.head.parameters(),lr=learning_rate)
    opt_base = torch.optim.SGD(model.base.parameters(), lr=learning_rate)
    
    n_trn = trn_x.shape[0]
    recon_n_trn = int(n_trn * supset_ratio)
    per_n_trn = n_trn - recon_n_trn
     
    recon_trn_x = trn_x[0:recon_n_trn]
    recon_trn_y = trn_y[0:recon_n_trn]
    per_trn_x = trn_x[recon_n_trn:]
    per_trn_y = trn_y[recon_n_trn:]
    
    loss_fn = nn.CrossEntropyLoss()
    
    recon_trn_gen = data.DataLoader(Dataset(recon_trn_x,recon_trn_y,train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    per_trn_gen = data.DataLoader(Dataset(per_trn_x,per_trn_y,train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)

    
    model.train(); model = model.to(device)
    
    #reconstruction phase
    for e in range(recon_epochs):
        # Training 
        recon_trn_gen_iter = recon_trn_gen.__iter__()
        for i in range(int(np.ceil(recon_n_trn/batch_size))):
            batch_x,batch_y = recon_trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            rep = model.base(batch_x)
            out_g = model.head(rep)
            loss_recon = loss_fn(out_g, batch_y.reshape(-1).long())
            opt_head.zero_grad()
            loss_recon.backward()
            opt_head.step()
            
    # personalzation phase
    for e in range(pers_epochs):
        # Training 
        per_trn_gen_iter = per_trn_gen.__iter__()
        for i in range(int(np.ceil(per_n_trn/batch_size))):
            batch_x,batch_y = per_trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            rep = model.base(batch_x)
            out_g = model.head(rep)
            loss_per = loss_fn(out_g, batch_y.reshape(-1).long())
            opt_base.zero_grad()
            loss_per.backward()
            opt_base.step()
            
    #Freese model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    
    return model

def train_fedrecon_mdl_(model, trainloader, learning_rate, batch_size,
                       recon_epochs, pers_epochs, recon_lr, pers_lr, supset_ratio,
                       print_per, dataset_name, device):
    
    opt_head = torch.optim.SGD(model.head.parameters(),lr=learning_rate)
    opt_base = torch.optim.SGD(model.base.parameters(), lr=learning_rate)

    n_trn = len(trainloader)
    recon_n_trn = int(n_trn * supset_ratio)
    
    range_recon = range(0, recon_n_trn)
    range_per = range(recon_n_trn, n_trn)
    
    loss_fn = nn.CrossEntropyLoss()
    
    model.train(); model = model.to(device)
    
    #reconstruction phase
    for e in range(recon_epochs):
        for i in (images, target) in enumerate(trainloader):
            if i not in range_recon: continue
            # compute output
            images = images.to(device)
            target = target.to(device)
            
            rep = model.base(images)
            out_g = model.head(rep)
            loss_recon = loss_fn(out_g, target)
            opt_head.zero_grad()
            loss_recon.backward()
            opt_head.step()
            
    # personalzation phase
    for e in range(pers_epochs):
        # Training 
        for i in (images, target) in enumerate(trainloader):
            if i in range_recon: continue
            # compute output
            images = images.to(device)
            target = target.to(device)
            
            rep = model.base(images)
            out_g = model.head(rep)
            loss_per = loss_fn(out_g, target)
            opt_base.zero_grad()
            loss_per.backward()
            opt_base.step()
            
    #Freese model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    
    return model