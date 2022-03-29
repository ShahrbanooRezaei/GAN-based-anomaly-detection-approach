""" 
Modified by Shahrbanoo Rezaei,  01/01/2021
Source code: "https://github.com/samet-akcay/ganomaly/tree/master/lib"

Using GAnomaly for Anomaly Detection in CAV
"""

"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing

##
def load_Train_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Returns:
        [type]: dataloader
    """
    ##
    ######## LOAD Train DATA SET
    root = opt.Train_dataset_root
    df = pd.read_csv(root, delimiter=' ', header=None)
    
    train = np.array(df)[:opt.n,:]
    
    
    m, n = train.shape 
    
    #Normalize
    scaler = preprocessing.MaxAbsScaler().fit(train)
    samples = scaler.transform(train)
    
    # Create Window
    seq_length = opt.isize2
    seq_step = opt.seq_sp
    num_signals = opt.isize1
    
    num_samples =  1+ ((samples.shape[0] - seq_length) // seq_step)
    aa = np.empty([num_samples, num_signals, seq_length ])
    
    for j in range(num_samples):
        for i in range(num_signals):
            aa[j, i, :] = samples[(j * seq_step):(j * seq_step + seq_length), i]
    
    
    TR = np.zeros((aa.shape[0],1, opt.isize1, opt.isize2))
    
    
    for i in range(aa.shape[0]):
        TR[i,0,:,:] = aa[i,:,:]
        
    Label = np.zeros((TR.shape[0],1))

    tensor_x = torch.Tensor(TR) # transform to torch tensor
    tensor_y = torch.Tensor(Label)
    
    dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=opt.batchsize,
                                                 shuffle=True,
                                                 num_workers=int(opt.workers),
                                                 drop_last=True,
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                 else lambda x: np.random.seed(opt.manualseed)))
    return dataloader, scaler

def load_Train_data_2(opt, scaler):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Returns:
        [type]: dataloader
    """
    ##
    ######## LOAD Train DATA SET
    root = opt.Train_dataset_root
    df = pd.read_csv(root, delimiter=' ', header=None)
    
    train = np.array(df)[:opt.n,:]
    
    
    m, n = train.shape 
    
    #Normalize
    scaler = preprocessing.MaxAbsScaler().fit(train)
    samples = scaler.transform(train)
    
    # Create Window
    seq_length = opt.isize2
    seq_step = opt.seq_sp
    num_signals = opt.isize1
    
    num_samples =  1+ ((samples.shape[0] - seq_length) // seq_step)
    aa = np.empty([num_samples, num_signals, seq_length ])
    
    for j in range(num_samples):
        for i in range(num_signals):
            aa[j, i, :] = samples[(j * seq_step):(j * seq_step + seq_length), i]
    
    
    TR = np.zeros((aa.shape[0],1, opt.isize1, opt.isize2))
    
    
    for i in range(aa.shape[0]):
        TR[i,0,:,:] = aa[i,:,:]
        
    Label = np.zeros((TR.shape[0],1))

    tensor_x = torch.Tensor(TR) # transform to torch tensor
    tensor_y = torch.Tensor(Label)
    
    dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    print('gi')
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=opt.batchsize,
                                                 shuffle=False,
                                                 num_workers=int(opt.workers),
                                                 drop_last=False,
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                 else lambda x: np.random.seed(opt.manualseed)))
    return dataloader


##
def load_Test_data(opt, scaler):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Returns:
        [type]: dataloader
    """
    ##
    ######## LOAD test DATA SET
    root = opt.Test_dataset_root
    df = pd.read_csv(root, delimiter=' ', header=None)
    
    
    test = np.array(df)
    m, n = test.shape # m=2000, n=4
    
    #Normalize
    if opt.scal == 1:
        scaler = preprocessing.MaxAbsScaler().fit(test) #????????????????????preprocessing.StandardScaler()
    samples = scaler.transform(test)
    
    # Create Window
    seq_length = opt.isize2
    seq_step = opt.isize2
    num_signals = opt.isize1
    
    num_samples = 1+ ((samples.shape[0] - seq_length) // seq_step)
    aa = np.empty([num_samples, num_signals, seq_length ])
    
    for j in range(num_samples):
        for i in range(num_signals):
            aa[j, i, :] = samples[(j * seq_step):(j * seq_step + seq_length), i]
    
    
    TR = np.zeros((aa.shape[0],1, opt.isize1, opt.isize2))
    
    
    for i in range(aa.shape[0]):
        TR[i,0,:,:] = aa[i,:,:]
        
    Label = np.zeros((TR.shape[0],1)) #????????????

    tensor_x = torch.Tensor(TR) # transform to torch tensor
    tensor_y = torch.Tensor(Label)
    
    dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=opt.batchsize,
                                                 shuffle=False,
                                                 num_workers=int(opt.workers),
                                                 drop_last=False,
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                 else lambda x: np.random.seed(opt.manualseed)))
    return dataloader

def load_val_data(opt, scaler):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Returns:
        [type]: dataloader
    """
    ##
    ######## LOAD Train DATA SET
    root = opt.Train_dataset_root
    df = pd.read_csv(root, delimiter=' ', header=None)
    
    val = np.array(df)[opt.n:opt.n+245*10000,:]
    
    
    m, n = val.shape 
    
    #Normalize
    if opt.scal == 1:
        scaler = preprocessing.MaxAbsScaler().fit(val)
    samples = scaler.transform(val)
    
    # Create Window
    seq_length = opt.isize2
    seq_step = opt.seq_sp
    num_signals = opt.isize1
    
    num_samples =  1+ ((samples.shape[0] - seq_length) // seq_step)
    aa = np.empty([num_samples, num_signals, seq_length ])
    
    for j in range(num_samples):
        for i in range(num_signals):
            aa[j, i, :] = samples[(j * seq_step):(j * seq_step + seq_length), i]
    
    
    TR = np.zeros((aa.shape[0],1, opt.isize1, opt.isize2))
    
    
    for i in range(aa.shape[0]):
        TR[i,0,:,:] = aa[i,:,:]
        
    Label = np.zeros((TR.shape[0],1))

    tensor_x = torch.Tensor(TR) # transform to torch tensor
    tensor_y = torch.Tensor(Label)
    
    dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=opt.batchsize,
                                                 shuffle=False,
                                                 num_workers=int(opt.workers),
                                                 drop_last=False,
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                 else lambda x: np.random.seed(opt.manualseed)))
    return dataloader