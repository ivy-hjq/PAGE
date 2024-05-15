from gym.envs.my_env.FL.utils.utils_libs import *
from gym.envs.my_env.FL.utils.utils_datasets import *
from gym.envs.my_env.FL.utils.utils_models import *
# Global parameters
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

def accuracy(output, target, topk=(1,)):
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

def get_acc_loss_(test_loader, model, dataset_name,device, w_decay = None):
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
            acc1, acc5 = accuracy(output, target, topk=(1,5))
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

# --- Helper functions

def set_client_from_params(mdl, params,device):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
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

# --- Train functions

def train_model(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name,device):
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
    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" %(e+1, acc_trn, loss_trn))
    model.train()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model,acc_trn,loss_trn




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
            
    # if (e+1) % print_per == 0:
    loss_trn, acc1_trn, acc5_trn = get_acc_loss_(train_loader, model, weight_decay)
    print("Epoch %3d, Training Acc1: %.4f Acc5:%.4f, Loss: %.4f" %(e+1, acc1_trn, acc5_trn, loss_trn))
    model.train()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model, acc1_trn, loss_trn