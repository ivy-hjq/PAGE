from .utils_libs import *
import logging
import numpy as np
np.random.seed(2023)
from numpy.core.fromnumeric import mean
import torch.utils.data as data
import torchvision.transforms as transforms
import random
random.seed(2023) 
from .datasets import CIFAR_truncated, ImageFolder_custom
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class DatasetObject:
    def __init__(self, dataset, data_path, n_client, rule, unbalanced_sgm=0, rule_arg=''):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        self.rule_arg = rule_arg
        self.ratio = 0.7
        rule_arg_str  = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%s_%s" %(self.dataset, self.n_client, self.rule, rule_arg_str)
        self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()
        
    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%s/%s' %(self.data_path, self.name)):
            # Get Raw data                
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trnset = torchvision.datasets.MNIST(root='%s/Raw' %self.data_path, 
                                                    train=True , download=True, transform=transform)
                tstset = torchvision.datasets.MNIST(root='%s/Raw' %self.data_path, 
                                                    train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trnset = torchvision.datasets.CIFAR10(root='%s/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%s/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
                
            if self.dataset == 'CIFAR100':
                print(self.dataset)
                # mean and std are validated here: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
                CIFAR_STD = [0.2673, 0.2564, 0.2762]
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ])

                valid_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ])
                # transform = transforms.Compose([transforms.ToTensor(),
                #                                 transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                #                                                      std=[0.2675, 0.2565, 0.2761])])
                trnset = torchvision.datasets.CIFAR100(root='%s' %self.data_path,
                                                      train=True , download=True, transform=train_transform)
                tstset = torchvision.datasets.CIFAR100(root='%s' %self.data_path,
                                                      train=False, download=True, transform=valid_transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'Imagenet200':
                print(self.dataset)
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                crop_scale = 0.08
                jitter_param = 0.4
                image_size = 224
                image_resize = 256

                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)),
                    transforms.ColorJitter(
                        brightness=jitter_param, contrast=jitter_param,
                        saturation=jitter_param),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
                valid_transform = transforms.Compose([
                    transforms.Resize(image_resize),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
                # transform = transforms.Compose([transforms.ToTensor(),
                #                                 transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                #                                                      std=[0.2675, 0.2565, 0.2761])])
                trnset = torchvision.datasets.ImageFolder(root='%s/train' %self.data_path, transform=train_transform)
                tstset = torchvision.datasets.ImageFolder(root='%s/val' %self.data_path,  transform=valid_transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=257015, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=100000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 224; self.height = 224; self.n_cls = 200;
            
            if self.dataset != 'emnist':
                
                trn_itr = trn_load.__iter__()
                tst_itr = tst_load.__iter__() 
                # labels are of shape (n_data,)
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()
                

                trn_x = trn_x.numpy(); trn_y = trn_y.numpy().reshape(-1,1)
                tst_x = tst_x.numpy(); tst_y = tst_y.numpy().reshape(-1,1)
            
            
            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)

                # load training labels
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1 # make first class 0

                # take first 10 classes of letters
                trn_idx = np.where(y_train < 10)[0]

                y_train = y_train[trn_idx]
                x_train = x_train[trn_idx]

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                # load test dataset
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)

                # load test labels
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1 # make first class 0

                tst_idx = np.where(y_test < 10)[0]

                y_test = y_test[tst_idx]
                x_test = x_test[tst_idx]
                
                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))
                
                # normalise train and test features

                trn_x = (x_train - mean_x) / std_x
                trn_y = y_train
                
                tst_x = (x_test  - mean_x) / std_x
                tst_y = y_test
                
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            # Shuffle Data
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]
            
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y
            
            
            ###
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            if self.unbalanced_sgm != 0:
                # Draw from lognormal distribution
                clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
                clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(trn_y)).astype(int)
                diff = np.sum(clnt_data_list) - len(trn_y)

                # Add/Subtract the excess number starting from first client
                if diff!= 0:
                    for clnt_i in range(self.n_client):
                        if clnt_data_list[clnt_i] > diff:
                            clnt_data_list[clnt_i] -= diff
                            break
            else:
                clnt_data_list = (np.ones(self.n_client) * n_data_per_clnt).astype(int)
            ###     
            
            if self.rule == 'Dirichlet':
                cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(trn_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

                clnt_x_all = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y_all = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
                clnt_data_list_copy=copy.deepcopy(clnt_data_list)
                while(np.sum(clnt_data_list)!=0):
                    curr_clnt = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    print('Remaining Data: %d' %np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        clnt_x_all[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y_all[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                
                clnt_x = [ np.zeros((int(self.ratio*clnt_data_list_copy[clnt__]), self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((int(self.ratio*clnt_data_list_copy[clnt__]), 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
                clnt_x_test = [ np.zeros((clnt_data_list_copy[clnt__]-int(self.ratio*clnt_data_list_copy[clnt__]), self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y_test = [ np.zeros((clnt_data_list_copy[clnt__]-int(self.ratio*clnt_data_list_copy[clnt__]), 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
                for clnt_i in range(self.n_client):
                    num_all = clnt_data_list_copy[clnt_i]
                    num_train = int(self.ratio * num_all)
                    clnt_x[clnt_i] = clnt_x_all[clnt_i][0:num_train]
                    clnt_y[clnt_i] = clnt_y_all[clnt_i][0:num_train]
                    clnt_x_test[clnt_i] = clnt_x_all[clnt_i][num_train-1:-1]
                    clnt_y_test[clnt_i] = clnt_y_all[clnt_i][num_train-1:-1]
                

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)
                clnt_x_test = np.asarray(clnt_x_test)
                clnt_y_test = np.asarray(clnt_y_test)
                
                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
            
            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
                assert len(trn_y)//100 % self.n_client == 0 
                # Only have the number clients if it divides 500
                # Perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x_all = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y_all = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx] # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client): # devices
                    for j in range(100): # class
                        clnt_x_all[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = trn_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        clnt_y_all[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = trn_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :] 
            
            
            elif self.rule == 'iid':
                                
                clnt_x_all = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y_all = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
            
                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x_all[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                    clnt_y_all[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                
                clnt_x = [ np.zeros((int(self.ratio*clnt_data_list_cum_sum[clnt__]), self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((int(self.ratio*clnt_data_list_cum_sum[clnt__]), 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
                clnt_x_test = [ np.zeros((clnt_data_list_cum_sum[clnt__]-int(self.ratio*clnt_data_list_cum_sum[clnt__]), self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y_test = [ np.zeros((clnt_data_list_cum_sum[clnt__]-int(self.ratio*clnt_data_list_cum_sum[clnt__]), 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
                for clnt_idx_ in range(self.n_client):
                    num_all = clnt_data_list_cum_sum[clnt_idx_]
                    num_train = int(self.ratio * num_all)
                    clnt_x[clnt_idx_] = clnt_x_all[clnt_idx_][0:num_train]
                    clnt_y[clnt_idx_] = clnt_y_all[clnt_idx_][0:num_train]
                    clnt_x_test[clnt_idx_] = clnt_x_all[clnt_idx_][num_train-1:-1]
                    clnt_y_test[clnt_idx_] = clnt_y_all[clnt_idx_][num_train-1:-1]
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)
                clnt_x_test = np.asarray(clnt_x_test)
                clnt_y_test = np.asarray(clnt_y_test)

            
            self.clnt_x = clnt_x; self.clnt_y = clnt_y
            self.clnt_x_test = clnt_x_test; self.clnt_y_test = clnt_y_test
            self.tst_x  = tst_x;  self.tst_y  = tst_y
            
            # Save data
            os.mkdir('%s/%s' %(self.data_path, self.name))
            
            np.save('%s/%s/clnt_x.npy' %(self.data_path, self.name), clnt_x)
            np.save('%s/%s/clnt_y.npy' %(self.data_path, self.name), clnt_y)

            np.save('%s/%s/clnt_x_test.npy' %(self.data_path, self.name), clnt_x_test)
            np.save('%s/%s/clnt_y_test.npy' %(self.data_path, self.name), clnt_y_test)

            np.save('%s/%s/tst_x.npy'  %(self.data_path, self.name),  tst_x)
            np.save('%s/%s/tst_y.npy'  %(self.data_path, self.name),  tst_y)

        else:
            print("Data is already downloaded in the folder.")
            self.clnt_x = np.load('%s/%s/clnt_x.npy' %(self.data_path, self.name), allow_pickle=True)
            self.clnt_y = np.load('%s/%s/clnt_y.npy' %(self.data_path, self.name), allow_pickle=True)
            self.clnt_x_test = np.load('%s/%s/clnt_x_test.npy' %(self.data_path, self.name), allow_pickle=True)
            self.clnt_y_test = np.load('%s/%s/clnt_y_test.npy' %(self.data_path, self.name), allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.tst_x  = np.load('%s/%s/tst_x.npy'  %(self.data_path, self.name), allow_pickle=True)
            self.tst_y  = np.load('%s/%s/tst_y.npy'  %(self.data_path, self.name), allow_pickle=True)
            
            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'Imagenet200':
                self.channels = 3; self.width = 224; self.height = 224; self.n_cls = 200;
            if self.dataset == 'fashion_mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
                
        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " %clnt + 
                  ', '.join(["%.3f" %np.mean(self.clnt_y[clnt]==cls) for cls in range(self.n_cls)]) + 
                  ', Amount:%d' %self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]
        
        
        print('Total Amount:%d' %count)
        print('--------')

        print("      Test: " + 
              ', '.join(["%.3f" %np.mean(self.tst_y==cls) for cls in range(self.n_cls)]) + 
              ', Amount:%d' %self.tst_y.shape[0])
        
def generate_syn_logistic(dimension, n_clnt, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False, iid_dat=False):
    
    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points
    
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)
    
    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_clnt)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' %np.sum(samples_per_user))
    
    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_clnt))
    data_y = list(range(n_clnt))

    mean_W = np.random.normal(0, alpha, n_clnt)
    B = np.random.normal(0, beta, n_clnt)

    mean_x = np.zeros((n_clnt, dimension))

    if not iid_dat: # If IID then make all 0s.
        for i in range(n_clnt):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))
    
    if iid_sol: # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))
    
    for i in range(n_clnt):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1,1)
    
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    print("data shape")
    print(data_x.shape)
    print(data_y.shape)
    return data_x, data_y
        
class DatasetSynthetic:
    def __init__(self, alpha, beta, theta, iid_sol, iid_data, n_dim, n_clnt, n_cls, avg_data, split_ratio, split_ratio_global, name_prefix):
        self.dataset = 'synt'
        self.name  = name_prefix + '_'
        self.name += '%d_%d_%d_%d_%f_%f_%f_%s_%s' %(n_dim, n_clnt, n_cls, avg_data,
                alpha, beta, theta, iid_sol, iid_data)

        data_path = 'Data'
        if (not os.path.exists('%s/%s/' %(data_path, self.name))):
            # Generate data
            print('Sythetize')
            data_x, data_y = generate_syn_logistic(dimension=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data, 
                                        alpha=alpha, beta=beta, theta=theta, 
                                        iid_sol=iid_sol, iid_dat=iid_data)  
            # data_x_test,data_y_test = generate_syn_logistic(dimension=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data_test, 
            #                             alpha=alpha, beta=beta, theta=theta, 
            #                             iid_sol=iid_sol, iid_dat=iid_data)
            num_local=int(avg_data*split_ratio_global)
            data_x,data_x_global=np.split(data_x,(num_local,),axis=1)
            data_y,data_y_global=np.split(data_y,(num_local,),axis=1)
            num_train=int(avg_data*split_ratio_global*split_ratio)
            print(num_train)
            data_x,data_x_test=np.split(data_x,(num_train,),axis=1)
            data_y,data_y_test=np.split(data_y,(num_train,),axis=1)
            os.mkdir('%s/%s/' %(data_path, self.name))
            np.save('%s/%s/data_x.npy' %(data_path, self.name), data_x)
            np.save('%s/%s/data_y.npy' %(data_path, self.name), data_y)
            np.save('%s/%s/data_x_test.npy' %(data_path, self.name), data_x_test)
            np.save('%s/%s/data_y_test.npy' %(data_path, self.name), data_y_test)
            np.save('%s/%s/data_x_global.npy' %(data_path, self.name), data_x_global)
            np.save('%s/%s/data_y_global.npy' %(data_path, self.name), data_y_global)
        else:
            # Load data
            print('Load')
            data_x = np.load('%s/%s/data_x.npy' %(data_path, self.name), allow_pickle=True)
            data_y = np.load('%s/%s/data_y.npy' %(data_path, self.name), allow_pickle=True)
            data_x_test = np.load('%s/%s/data_x_test.npy' %(data_path, self.name), allow_pickle=True)
            data_y_test = np.load('%s/%s/data_y_test.npy' %(data_path, self.name), allow_pickle=True)
            data_x_global = np.load('%s/%s/data_x_global.npy' %(data_path, self.name), allow_pickle=True)
            data_y_global = np.load('%s/%s/data_y_global.npy' %(data_path, self.name), allow_pickle=True)
        for clnt in range(n_clnt):
            print(', '.join(['%.4f' %np.mean(data_y[clnt]==t) for t in range(n_cls)]))
            print(', '.join(['%.4f' %np.mean(data_y_test[clnt]==t) for t in range(n_cls)]))
            print(', '.join(['%.4f' %np.mean(data_y_global[clnt]==t) for t in range(n_cls)]))
            

        self.clnt_x = data_x
        self.clnt_y = data_y
        self.clnt_x_test = data_x_test
        self.clnt_y_test = data_y_test
        self.clnt_x_global = data_x_global
        self.clnt_y_global = data_y_global

        self.tst_x = np.concatenate(self.clnt_x_global, axis=0)
        self.tst_y = np.concatenate(self.clnt_y_global, axis=0)
        self.n_client = len(data_x)
        print(self.clnt_x.shape)

# Original prepration is from LEAF paper...
# This loads Shakespeare dataset only.
# data_path/train and data_path/test are assumed to be processed
# To make the dataset smaller,
# We take 2000 datapoints for each client in the train_set

class ShakespeareObjectCrop:
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')
        
        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        # Ignore groups information, combine test cases for different clients into one test data
        # Change structure to DatasetObject structure
        
        self.users = users
        
        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))
        
        tst_data_count = 0
        
        for clnt in range(self.n_client):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[clnt]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[clnt]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[clnt]]['y'])[start:start+crop_amount]
            
        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))
        
        tst_data_count = 0
        for clnt in range(self.n_client):
            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[clnt]]['x'])-curr_amount)
            self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['x'])[start:start+curr_amount]
            self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['y'])[start:start+curr_amount]
            
            tst_data_count += curr_amount
        
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)
        
        # Convert characters to numbers
        
        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)
        
        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)
        
        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))
        

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            
            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)
                
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        
        
        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))
                
        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)
        
        
class ShakespeareObjectCrop_noniid:
    def __init__(self, data_path, dataset_prefix, n_client=100, crop_amount=7000, tst_ratio=7, split_ratio=0.7, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.        
        # Change structure to DatasetObject structure
        
        self.users = users 

        tst_data_count_per_clnt = (crop_amount//tst_ratio)
        # Group clients that have at least crop_amount datapoints
        arr = []
        for clnt in range(len(users)):
            if (len(np.asarray(train_data[users[clnt]]['y'])) > crop_amount 
                and len(np.asarray(test_data[users[clnt]]['y'])) > tst_data_count_per_clnt):
                arr.append(clnt)

        # choose n_client clients randomly
        self.n_client = n_client
        np.random.seed(rand_seed)
        np.random.shuffle(arr)
        self.user_idx = arr[:self.n_client]
          
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))
        self.clnt_x_test = list(range(self.n_client))
        self.clnt_y_test = list(range(self.n_client))
        
        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[idx]]['x'])-crop_amount)
            split = int(start+crop_amount*split_ratio)
            self.clnt_x[clnt] = np.asarray(train_data[users[idx]]['x'])[start:split]
            self.clnt_x_test[clnt] = np.asarray(train_data[users[idx]]['x'])[split:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[idx]]['y'])[start:split]
            self.clnt_y_test[clnt] = np.asarray(train_data[users[idx]]['y'])[split:start+crop_amount]

            
        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))
        
        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            
            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[idx]]['x'])-curr_amount)
            self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['x'])[start:start+curr_amount]
            self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['y'])[start:start+curr_amount]
            tst_data_count += curr_amount
            
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        self.clnt_x_test = np.asarray(self.clnt_x_test)
        self.clnt_y_test = np.asarray(self.clnt_y_test)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)
        
        # Convert characters to numbers
        
        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)
        self.clnt_x_test_char = np.copy(self.clnt_x_test)
        self.clnt_y_test_char = np.copy(self.clnt_y_test)
        
        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)
        
        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))
        
        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            
            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)
                
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.clnt_x_test = list(range(len(self.clnt_x_test_char)))
        self.clnt_y_test = list(range(len(self.clnt_x_test_char)))

        for clnt in range(len(self.clnt_x_test_char)):
            clnt_list_x = list(range(len(self.clnt_x_test_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_test_char[clnt])))
            
            for idx in range(len(self.clnt_x_test_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_test_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_test_char[clnt][idx]))).reshape(-1)

            self.clnt_x_test[clnt] = np.asarray(clnt_list_x)
            self.clnt_y_test[clnt] = np.asarray(clnt_list_y)
        
        
        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_y_char)))
                
        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)
    
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()
            
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == 'Imagenet200':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
        
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
                
        elif self.name == 'shakespeare':
            
            self.X_data = data_x
            self.y_data = data_y
                
            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()
            
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y
        
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == 'Imagenet200':
            img = self.X_data[idx]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
            
        elif self.name == 'shakespeare':
            x = self.X_data[idx]
            y = self.y_data[idx] 
            return x, y

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def _data_transforms_cifar(datadir):
    if "cifar100" in datadir:
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]
    else:
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def _data_transforms_imagenet(datadir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if "tiny" in datadir:
        train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
    else:
        crop_scale = 0.08
        jitter_param = 0.4
        image_size = 224
        image_resize = 256    
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(image_resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return train_transform, valid_transform


def load_data(datadir):
    if 'cifar' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR_truncated
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_custom
    train_ds = dl_obj(datadir, train=True, download=True, transform=train_transform)
    test_ds = dl_obj(datadir, train=False, download=True, transform=test_transform)

    y_train, y_test = train_ds.target, test_ds.target

    return (y_train, y_test)
def partition_data(datadir, partition, n_nets, alpha, unbalanced_sgm, test_client_number):
    logging.info("*********partition data***************")
    y_train, y_test = load_data(datadir)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    class_num = len(np.unique(y_train))

    test_net_dataidx_map={}
    # test net_dataid_map
    if test_client_number>0:
        test_total_num = n_test
        test_idxs = np.random.permutation(test_total_num)
        test_batch_idxs =  np.array_split(test_idxs, test_client_number) 
        test_net_dataidx_map = {i: test_batch_idxs[i] for i in range(test_client_number)}
    
    # train net_dataid_map
    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    elif partition == "hetero" and unbalanced_sgm>0:
        print("hetero",alpha)
        print("unbalanced",unbalanced_sgm)
        total_num = n_train
        net_dataidx_map = {}
        idxs = np.random.permutation(total_num)
        num_samples_per_client = int(total_num / n_nets)
        client_sample_nums = np.random.lognormal(mean=np.log(num_samples_per_client), sigma=unbalanced_sgm, size=n_nets)
        client_sample_nums = ((client_sample_nums/np.sum(client_sample_nums))*total_num).astype(int)
       
        diff = np.sum(client_sample_nums) - total_num # diff <= 0
        
        # Add/Subtract the excess number starting from first client
        if diff!=0:
            for cid in range(n_nets):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break
        
        class_priors = np.random.dirichlet(alpha=[alpha]*class_num,size=n_nets)
        prior_cumsum = np.cumsum(class_priors, axis=1)
        idx_k = [np.where(y_train==k)[0] for k in range(class_num)]
        class_amount = [len(idx_k[k]) for k in range(class_num)]
        client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in range(n_nets)]
        
        while np.sum(client_sample_nums) != 0:
            curr_cid = np.random.randint(n_nets)
            # If current node is full resample a client
            
            print('Remaining Data: %d' % np.sum(client_sample_nums))
            if client_sample_nums[curr_cid] <= 0:
                continue
            client_sample_nums[curr_cid] -= 1
            curr_prior = prior_cumsum[curr_cid]
            while True:
                curr_class = np.argmax(np.random.uniform() <= curr_prior)
                if class_amount[curr_class] <= 0:
                    continue
                
                class_amount[curr_class] -= 1
                client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                    idx_k[curr_class][class_amount[curr_class]]

                break
        net_dataidx_map={cid: client_indices[cid] for cid in range(n_nets)}
            
    elif partition == "hetero" and unbalanced_sgm==0:
        min_size = 0
        K = class_num
        N = n_train
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "shards":
        total_num = n_train
        K = class_num
        # n_shards = int(alpha * K)
        n_shards = alpha #the number of total shards
        # n_clients_shards = np.ones(n_nets)*int(n_shards/n_nets) #the number of each client's shards
        n_per_client =np.ones(n_nets)*int(total_num / n_nets)
        num_cumsum = np.cumsum(n_per_client).astype(int)
        # ratio = (K/n_nets)/n_shards
        net_dataidx_map = {}
        k_list=[]
        idx_batch = [[] for _ in range(n_nets)]
        # order the idx by labels
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            k_list.append(idx_k)
        k_list = np.array(k_list) 
        k_list = k_list.flatten() 
        # split the total idx to shards
        shards_idx = np.array_split(k_list, n_shards)
        np.random.shuffle(shards_idx)
        k_list = np.array(shards_idx).flatten()
        idx_batch=np.split(k_list,num_cumsum)
        
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j].tolist()
            
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return class_num, net_dataidx_map, test_net_dataidx_map, traindata_cls_counts

def data_split(full_list,ratio,shuffle=False):
    """
   randomly split full_list by ratio to two sublist 
    """
    n_total=len(full_list)
    offset=int(n_total*ratio)
    if n_total == 0 or offset<1 or full_list==None:
        return full_list,None
    if shuffle:
        random.shuffle(full_list)
    sublist_1=full_list[:offset]
    sublist_2=full_list[offset:]
    return sublist_1, sublist_2

# for centralized training
def get_dataloader(datadir, train_bs, test_bs, dataidxs=None):
    if 'cifar' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR_truncated
        workers=0
        persist=False       
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_custom
        workers=0
        persist=False
    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=train_transform, download=True)
    test_ds = dl_obj(datadir, train=False, transform=test_transform, download=True)
    
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=persist)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True, num_workers=workers, persistent_workers=persist)

    return train_dl, test_dl

def get_local_dataloader(datadir, train_bs, test_bs, dataidxs=None, ratio=0.8, train_dataset=True):
    if 'cifar' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR_truncated
        workers=0
        persist=False
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_custom
        workers=0
        persist=False
    train_dataidxs,test_dataidxs = data_split(dataidxs,ratio,shuffle=True)
    train_ds = dl_obj(datadir, dataidxs=train_dataidxs, train=train_dataset, transform=train_transform, download=True)
    test_ds = dl_obj(datadir, dataidxs=test_dataidxs, train=train_dataset, transform=test_transform, download=True)
    
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=persist)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True, num_workers=workers, persistent_workers=persist)

    return train_dl, test_dl,train_ds

def load_partition_data(data_dir, partition_method, partition_alpha, unbalanced_sgm, client_number, batch_size,ratio,test_client_number,recon_ratio):
    class_num, net_dataidx_map, test_net_dataidx_map, traindata_cls_counts = partition_data(data_dir, partition_method, client_number, partition_alpha, unbalanced_sgm,test_client_number)

    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    
    #get train dataset loader and test dataset loader
    train_data_global, test_data_global = get_dataloader(data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_local_dataset = dict()
    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        # print("client: %d, dataidxs:" % (client_idx))
        # print(dataidxs)
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local ,train_dataset = get_local_dataloader(data_dir, batch_size, batch_size, dataidxs,ratio, train_dataset=True)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        train_local_dataset[client_idx] = train_dataset
    
    # get test local dataset for fedrecon
        test_data_local_num_dict = dict()
        recon_data_local_dict = dict()
        test_test_data_local_dict = dict() 
    if test_client_number > 0:
        test_test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(test_client_number)])
        
        for test_client_idx in range(test_client_number):
            test_dataidxs = test_net_dataidx_map[test_client_idx]
            # print("client: %d, dataidxs:" % (test_client_idx))
            # print(test_dataidxs)
            test_local_data_num = len(test_dataidxs)
            test_data_local_num_dict[test_client_idx] = test_local_data_num
            logging.info("test_client_idx = %d, local_sample_number = %d" % (test_client_idx, test_local_data_num))
            recon_data_local, test_test_data_local, test_dataset = get_local_dataloader(data_dir, batch_size, batch_size, test_dataidxs,recon_ratio, train_dataset=False)
            logging.info("testclient_idx = %d, batch_num_recon_local = %d, batch_num_test_local = %d" % (
                test_client_idx, len(recon_data_local), len(test_test_data_local)))
            recon_data_local_dict[test_client_idx] = recon_data_local
            test_test_data_local_dict[test_client_idx] = test_test_data_local
    return train_data_global,test_data_global, train_data_local_dict, test_data_local_dict, recon_data_local_dict, test_test_data_local_dict

class ImagenetObjectCrop_noniid:
    def __init__(self, dataset, data_path, n_client, rule='hetero', unbalanced_sgm=0, rule_arg=0.3,split_ratio=0.8,batch_size=32, test_client_number = 0,test_recon_ratio=0.7*0.5):
        self.n_client = n_client
        self.dataset = dataset
        self.rule = rule
        self.name = "%s_%d_%s_%s" %(self.dataset, self.n_client, self.rule, rule_arg)
        self.train_loader,self.test_loader,self.clnt_train_loader,self.clnt_test_loader,self.test_clnt_recon_loader, self.test_clnt_test_loader=load_partition_data(data_dir=data_path,partition_method=rule,partition_alpha=rule_arg,unbalanced_sgm=unbalanced_sgm,client_number=n_client,batch_size=batch_size,ratio=split_ratio, test_client_number=test_client_number,recon_ratio=test_recon_ratio)
 