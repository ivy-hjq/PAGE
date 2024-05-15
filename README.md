# PAGE: Equilibrate Personalization and Generalization in Federated Learning

This is implementation of PAGE: Equilibrate Personalization and Generalization in Federated Learning.

### Requirements

Please install the required packages. The code is compiled with Python 3.8 dependencies in a virtual environment via

``pip install -r requirements.txt``
We construct our own FL enviroments ``./my_envs`` for PAGE and Dap-FL to implement reinforcement learning argorithms. In order to use a standard API of [OpenAI Gym](https://github.com/openai/gym),
please remove ``./my_env`` to ``gym/envs``, append the contents of ``gym/envs/my_env/__init__.py`` to the end of ``gym/envs/__init__.py`` and delete ``gym/envs/my_env/__init__.py``

### Instructions

Codes to run PAGE, Dap-FL as well as other baseline methods (FedAvg, FedProx, SCAFFOLD, FedDyn, pFedMe, Ditto, FedALA, Fed-ROD and FedRECON) with the Synthetic dataset, CIFAR100, Tiny-imagenet-200 and Shakespeare are given in ``./PAGE&baseline(Dap-FL)/PAGE/``, ``./PAGE&baseline(Dap-FL)/Dap`` and ``./Baselines/main``, respectively.

Before running codes of PAGE and baselines with tiny-imagenet-200 and shakespeare, it is needed to run ``./Data/run.sh`` to generate tiny-imagenet-200 dataset and run ``./LEAF/shakespeare/preprocess.sh`` to generate shakespeare dataset, respeactively.

Please set the parameters in ``my_env/FL/synthetic_FL.py``, ``my_env/FL/cifar100_FL.py``, ``my_env/FL/imagenet200_FL.py``, and ``my_env/FL/shakespeare_FL.py`` to control the FL setting in PAGE and Dap-FL with the Synthetic dataset, CIFAR100, Tiny-imagenet-200 and Shakespeare, respectively.
Please set the parameters in  ``./Baselines/main/main_synthetic.py``, ``./Baselines/main/main_cifar100.py``, ``./Baselines/main/main_tiny-imagenet200.py``, and ``./Baselines/main/main_shakespeare.py`` to control other baselines' FL setting with the Synthetic dataset, CIFAR100, Tiny-imagenet-200 and Shakespeare, respectively.

#### Generate IID and Non-IID on various datasets:

##### Generate Shakespeare

Generate Shakespeare with non-iid used in PAGE to run

```
./LEAF/shakespeare/preprocess.sh -s niid --sf 1.0 -k 8000 -t sample -tf 0.8 --smplseed 1685439952 --spltseed 1685439964
```

Please reference ``./LEAF/README.md`` to know more about the details of generating shakespeare dataset.

##### Generate Cifar100

CIFAR100 IID, 100 partitions, balanced data, local training and testing sets on a 7:3 scale.

```
data_obj = ImagenetObjectCrop_noniid(dataset='CIFAR100', n_client=100, rule='homo', unbalanced_sgm=0, split_ratio=0.7, test_client_number=0)
```

CIFAR100 Dirichlet (0.3), 100 partitions, balanced data, local training and testing sets on a 7:3 scale.

```
data_obj = ImagenetObjectCrop_noniid(dataset='CIFAR100', n_client=100, rule='hetero', rule_arg=0.3, unlalanced_sgm=0, split_ratio=0.7, test_client_number=0)
```

CIFAR100 Dirichlet (0.3), 100 partitions, unbalanced (0.1) data, local training and testing sets on a 7:3 scale.

```
data_obj = ImagenetObjectCrop_noniid(dataset='CIFAR100', n_client=100, rule='hetero', ule_arg=0.3, unlalanced_sgm=0.1, split_ratio=0.7, test_client_number=0)
```

##### Generate Tiny-imagent-200

Generate Tiny-imagenet-200 to run

```
./Data/run.sh
```

Tiny-imagenet-200 IID, 100 partitions, balanced data, local training and testing sets on a 7:3 scale.

```
data_obj = ImagenetObjectCrop_noniid(dataset='Imagenet200', n_client=100, rule='homo', unbalanced_sgm=0, split_ratio=0.7, test_client_number=0)
```

Tiny-imagenet-200 Dirichlet (0.3), 100 partitions, balanced data, local training and testing sets on a 7:3 scale.

```
data_obj = ImagenetObjectCrop_noniid(dataset='Imagenet200', n_client=100, rule='hetero', rule_arg=0.3, unlalanced_sgm=0, split_ratio=0.7, test_client_number=0)
```

Tiny-imagenet-200 Dirichlet (0.3), 100 partitions, unbalanced (0.1) data, local training and testing sets on a 7:3 scale.

```
data_obj = ImagenetObjectCrop_noniid(dataset='Imagenet200', n_client=100, rule='hetero', ule_arg=0.3, unlalanced_sgm=0.1, split_ratio=0.7, test_client_number=0)
```

##### Generate Synthetic dataset

Synthetic dataset Non-IID, 100 partitions, used in PAGE.

```
data_obj = DatasetSynthetic(alpha=0, beta=1, theta=0, iid_sol=True, iid_data=False, n_dim=30, n_clnt=100, n_cls=30, avg_data=375, split_ratio=0.7, split_ratio_global=0.8,name_prefix='syn_alpha-0_beta-1_theta0')
```
