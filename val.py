import os
import numpy as np
import torch

from LossUtils import *
from Structure.googlelike import GoogleLin
from Structure.KPP_Net import KPPNet
from DataSets.loadData import load_dataSet
import Config as config
from sklearn.metrics import r2_score
kppnet = KPPNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config.use_our_dataSets:
    train_db, val_x, val_y = load_dataSet(True, mix_all=config.mix_all_data)
else:
    train_db, val_x, val_y = load_dataSet(True, None, None, mix_all=config.mix_all_data)
val_x = val_x.to(device)
val_y = val_y.to(device)
out_feature = 1

if config.use_physics:
    out_feature = 3
if config.use_our_model:
    model = GoogleLin(5, out_feature, config.inception_layers, True)
else:
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.LeakyReLU(),
        nn.Linear(10, 10),
        nn.LeakyReLU(),
        nn.Linear(10, 5),
        nn.LeakyReLU(),
        nn.Linear(5, out_feature)
    )
model = model.to(device)
criterion = nn.MSELoss()
check_point = torch.load("output/best_0.627_google_4_layers_oriDataSet_mix.pth")# output/OriDataSet/mix/best_0.634_google_4_layers_oriDataSet_mix.pth
model.load_state_dict(check_point)
model.eval()
out = model(val_x)
loss = criterion(out[:, 0], val_y[:, 0])
vloss = torch.sqrt(loss).item()
# r2_fun = R2_loss()
t_out = out[:, 0].cpu().detach().numpy()
t_val = val_y[:, 0].cpu().detach().numpy()

# ri = 10 ** val_x[:, 4].cpu()
# kout = kppnet(val_x[:, 4].cpu())
# kout = kout.detach().numpy()
r2loss = r2_score(t_val, t_out)# r2_fun(out[:, 0], val_y[:, 0]).item()

print(vloss)
print(r2loss)
