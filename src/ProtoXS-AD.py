# Utils.py
import os
import pandas
import numpy as np
import time
import sys
import wandb
from models import *
from datasets import *
from utils import *
from transformations import * 
import torch
import matplotlib.pyplot as plt
device = torch.device("cuda")
from torch.utils.data import random_split

if __name__ == "__main__":
    dataset = sys.argv[1]
    num_prototypes_per_class = int(sys.argv[2]) 
    seed = int(sys.argv[3]) 
    if dataset == "UMD":
        _, train_data, test_data, test_labels, real_labels = load_dataset("UMD", 1, 0)
        normality = 1
        reverse = 0
        transformations = UMDTransformations()
        n_transforms = 9
        _, seq_len, n_features = train_data.shape
    elif dataset == "Yorkshire":
        dma = int(sys.argv[4])
        day = int(sys.argv[5])
        percentile = float(sys.argv[6])
        print(dma, day, percentile)
        train_data, test_data, test_labels = load_yorkshire(dma, day, percentile)
        seq_len = 17
        n_features = 1
        transformations = TransformationsYorkshire()
        n_transforms = 4
    elif dataset == "Temperature":
        source = str(sys.argv[4])
        train_data, test_data, test_labels, _ = load_temperatures(source)
        seq_len = 12
        n_features = 1
        transformations = TransformationsTemperature()
        n_transforms = 5
    name = "ProtoXS-AD_" + str(dataset) 
    train_data = train_data.reshape(train_data.shape[0], seq_len, n_features)
    test_data = test_data.reshape(test_data.shape[0], seq_len, n_features)

    
    torch.set_num_threads(1)
    device = torch.device("cuda")
    set_seed(seed)

    X_train = torch.from_numpy(train_data).float()
    X_test  = torch.from_numpy(test_data).float()
    
    full_train_dataset = SelfDataset(X_train)
    test_dataset = SelfDataset(X_test)

    input_dims = train_data[0].shape[1]
    
    # Crear un modelo autoencoder basado en transformer
    set_seed(seed)

    model = Experiment(input_dims = n_features, latent_dims = 320, seq_len = seq_len, n_transforms = n_transforms, train_data = full_train_dataset, val_data = full_train_dataset, test_data = test_dataset, transformations = transformations, num_prototypes_per_class = num_prototypes_per_class).to(device)
    
    start = time.time()
    model.train()
    set_seed(seed)
    model.train_model(max_epochs=1000)
    
    model.eval()
    set_seed(seed)
    scores = model.compute_scores()
    roc_post, pr_post = metrics(scores, test_labels)

    fig, proto_series = model.plot_prototypes_input(head="explainer", feature=0, ncols=10)

    test_loader = DataLoader(SelfDataset(test_data), batch_size=1, shuffle=False)
    mu_anom = anomalous_latents(test_loader, model)
    nearest_match = nearest_prototypes_global(mu_anom, model)[0]

    l1, l2, dt, l1_std, l2_std, dt_std = compute_distances_to_nearest_prototypes(model.test_loader, nearest_match, proto_series)

    end = time.time()

    

    wandb.log({
           "roc_post": roc_post,
           "pr_post": pr_post,
           "l1": l1,
           "l1_std": l1_std,
           "l2": l2,
           "l2_std": l2_std,
           "dtw": dt,
           "dtw_std": dt_std,
           "time": (end-start)})

  
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
    

