import os
import numpy as np
import torch

from LossUtils import *
from Structure.googlelike import GoogleLin
from Structure.OriginalNet import OriNet
from DataSets.loadData import load_dataSet
import Config as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config.use_our_dataSets:
    train_db, val_x, val_y = load_dataSet(True, mix_all=config.mix_all_data)
else:
    train_db, val_x, val_y = load_dataSet(True, None, None, mix_all=config.mix_all_data)
val_x = val_x.to(device)
val_y = val_y.to(device)

out_feature = 1
# 使用物理约束，调整输出为Kt,A,B
if config.use_physics:
    out_feature = 3
# 模型切换
if config.use_our_model:
    model = GoogleLin(5, out_feature, config.inception_layers, config.use_Kpp)
else:
    model = OriNet()
    # model = nn.Sequential(
    #     nn.Linear(5, 10),
    #     nn.LeakyReLU(),
    #     nn.Linear(10, 10),
    #     nn.LeakyReLU(),
    #     nn.Linear(10, 5),
    #     nn.LeakyReLU(),
    #     nn.Linear(5, out_feature)
    # )
model = model.to(device)
# optim = torch.optim.Adam(model.parameters())
optim = torch.optim.NAdam(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)
criterion = nn.MSELoss()
best_loss = 100
best_fileName = "best.pth"

if config.load_static != "":
    print("加载模型微调")
    model.load_state_dict(torch.load(config.load_static))

for epoch in range(config.epoch_nums):
    train_all_loss = 0
    sample_count = 0
    for step, data in enumerate(train_db):
        x = data[0].to(device)
        y = data[1].to(device)

        optim.zero_grad()
        out = model(x)
        if config.use_physics:
            if not config.use_our_dataSets:
                print("数据集不支持该物理损失")
                exit()
            Kt = out[:, 0]
            A = out[:, 1]
            B = out[:, 2]
            T = B / torch.log(torch.abs(torch.pow(10, Kt) / A))
            tloss = criterion(T, y[:, 1])
            ktloss = criterion(Kt, y[:, 0]) * 10
            loss = tloss + ktloss
        else:
            if config.use_our_dataSets:
                y = y[:, 0]
                y = y.unsqueeze(1)
            loss = criterion(out, y)

        loss.backward()
        optim.step()
        train_all_loss += loss.item() * y.size(0)
        sample_count += y.size(0)
    train_loss_avg = np.sqrt(train_all_loss / sample_count)
    if config.open_cosine:
        scheduler.step()

    model.eval()
    with torch.no_grad():
        out = model(val_x)
        if config.use_physics or config.use_our_dataSets:
            loss = criterion(out[:, 0], val_y[:, 0])
        else:
            loss = criterion(out, val_y)
        vloss = torch.sqrt(loss).item()
        if best_loss > vloss:
            best_loss = vloss
            if os.path.exists("output/" + best_fileName):
                os.remove("output/" + best_fileName)

            best_fileName = ("best_%.3f_" % best_loss) + ("google_%d_layers_" % config.inception_layers if config.use_our_model else "ori_") + ("myDataSet" if config.use_our_dataSets else "oriDataSet") + ("_mix" if config.mix_all_data else "") + ("_physics" if config.use_physics else "") + ("_kpp" if config.use_Kpp else "") + ".pth"

            torch.save(model.state_dict(), "output/" + best_fileName)
        if epoch % 199 == 0:
            print("epoch: %d tloss: %.5f vloss: %.5f best: %.5f" % (epoch, train_loss_avg, vloss, best_loss))

    model.train()







