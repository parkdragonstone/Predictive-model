import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
import scipy.signal as signal
import time
import warnings
warnings.filterwarnings('ignore')
import copy
import datetime
from data_preprocessing_revise import data_processing, split_train_test, input_target,input_target_grouping
from model import CustomDataset, ResNet10

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(device)

file_list = [i.replace('\\','/') for i in sorted(glob(f"c3d/*txt"))]
file_name = []
for i in file_list:
    file, _, _ = os.path.basename(i).split('.')
    file_name.append(file)

axis_num = 3
FORCE, ANGLE, JOINT = data_processing(file_list, file_name)
player_trial, train, test = split_train_test(JOINT, random = 5)

inputs_sc, targets = input_target(JOINT, FORCE, train, test,joint_cut_off = 13.4, force_cut_off = 60)
input_target_sc_np = input_target_grouping(inputs_sc, targets)


data_set_np = input_target_sc_np
data_set_name = 'standard'
epochs = 1000
learning_rate = 0.001

rear_time = 0
lead_time = 0
total_time_start = time.time()

k_fold_avg_score = {
    'lead' : {
        'r_x' : 0,
        'r_y' : 0,
        'r_z' : 0,
        'loss' : 0},
    
    'rear' : {
        'r_x' : 0,
        'r_y' : 0,
        'r_z' : 0,
        'loss' : 0},
}


for fp in ['lead','rear']:
    start = time.time()
    best_loss = float('inf')
    kfolds = KFold(n_splits=9, shuffle=True, random_state=6)
    
    avg_r2_x = []
    avg_r2_y = []
    avg_r2_z = []
    avg_loss = []
    
    for fold, (t, v) in enumerate(kfolds.split(train)):
        if fold == 1:
            break
        tr = list(np.array(train)[t])
        va = list(np.array(train)[v])
        k_fold_since = time.time()
        train_sessions = list(inputs_sc['train'][fp])
        trn = []
        val = []
        for session in train_sessions:
            if session.split('_')[0] in tr:
                trn.append(session)
            elif session.split('_')[0] in va:
                val.append(session)
                
        train_dataset = {
            'x': torch.FloatTensor(np.concatenate([data_set_np['train'][fp]['input'][v] for v in trn], axis=0)), 
            'y': torch.FloatTensor(np.concatenate([data_set_np['train'][fp]['target'][v] for v in trn], axis=0))
            } 
        train_loader = DataLoader(CustomDataset(train_dataset['x'], train_dataset['y']), batch_size=32, shuffle=True)

        val_dataset = {
                        'x' : {v : torch.FloatTensor(data_set_np['train'][fp]['input'][v]) for v in val},
                        'y' : {v : torch.FloatTensor(data_set_np['train'][fp]['target'][v]) for v in val},    
                    }
        val_loader = {x : DataLoader(CustomDataset(val_dataset['x'][x], val_dataset['y'][x]), batch_size = len(val_dataset['x'][x])) for x in val}
        
        model = ResNet10(64, 128, 0.2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
        
        k_fold_r2_x = 0
        k_fold_r2_y = 0
        k_fold_r2_z = 0
        k_fold_val_loss = float('inf')
        min_val_loss = float('inf')
        n_total_steps = len(train_loader)

        for epoch in range(epochs):
            total_train_loss = 0
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            total_train_loss = total_train_loss/n_total_steps
            
            model.eval()
            val_r2 = []
            val_rmse = []
            val_losses = []
            val_N = len(val_loader)
            
            with torch.no_grad():
                for v in val_loader:
                    for inputs, targets in val_loader[v]:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_losses.append(loss.item())
                        
                        r2_x = r2_score(targets.cpu()[:,0], outputs.cpu()[:,0])
                        r2_y = r2_score(targets.cpu()[:,1], outputs.cpu()[:,1])
                        r2_z = r2_score(targets.cpu()[:,2], outputs.cpu()[:,2])
                        val_r2.append(np.array([r2_x, 
                                                r2_y,
                                                r2_z,
                                                ]))
                             
            val_r2 = sum(val_r2)/val_N
            val_rmse = sum(val_rmse)/val_N
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)
            
            val_r2_x = val_r2[0]
            val_r2_y = val_r2[1]
            val_r2_z = val_r2[2]
            
            if (epoch+1) % 2 == 0: 
                print(f"Fold [{fold + 1}] | Epoch {epoch+1}/{epochs} | Train Loss: {total_train_loss:.4f} | Val Loss : {val_loss:.4f} | R2 : X = {val_r2_x:.4f}, Y = {val_r2_y:.4f}, Z = {val_r2_z:.4f}")
            
            if k_fold_val_loss > val_loss:
                k_fold_r2_x = val_r2_x
                k_fold_r2_y = val_r2_y
                k_fold_r2_z = val_r2_z
                k_fold_loss = val_loss
                
            if best_loss > val_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)
                torch.save(model, f"model/ResNet2-re-{fp}-best-model.pt")
                torch.save(model.state_dict(), f'model/ResNet2-re-{fp}-best-model-parameters.pt')

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                early_stop_count = 0
                
            else:
                early_stop_count += 1
            
            if early_stop_count >= 10:
                print(f"Early stopping | Fold [{fold + 1}] | Epoch {epoch+1}/{epochs}")
                break
            
        avg_r2_x.append(k_fold_r2_x)
        avg_r2_y.append(k_fold_r2_y)
        avg_r2_z.append(k_fold_r2_z)
        avg_loss.append(k_fold_loss)
        
    avg_r2_x = np.mean(avg_r2_x)
    avg_r2_y = np.mean(avg_r2_y)
    avg_r2_z = np.mean(avg_r2_z)
    avg_loss = np.mean(avg_loss)
    
    k_fold_avg_score[fp]['r_x'] = avg_r2_x
    k_fold_avg_score[fp]['r_y'] = avg_r2_y
    k_fold_avg_score[fp]['r_z'] = avg_r2_z
    k_fold_avg_score[fp]['loss'] = avg_loss
    
    end = time.time()
    sec = (end - start)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")
    if fp == 'lead':
        lead_time = result_list[0]
    elif fp == 'rear':
        rear_time = result_list[0]

total_time_end = time.time()
total_sec = (total_time_end - total_time_start)
time_list = str(datetime.timedelta(seconds=total_sec)).split(".")

print(k_fold_avg_score)
print(f"total training time : {time_list[0]}")
print(f"lead training time : {lead_time}")
print(f"rear training time : {rear_time}")
