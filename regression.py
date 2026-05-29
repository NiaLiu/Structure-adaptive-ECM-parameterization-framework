
# Low, high norm: 0.        0, 281826.8101 
# Low, high norm: 1.12e-11, 32.486766 

# pip install torch numpy
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

# data
excluded = False
model = 'a'       #option: 'a' ,'b'
choose = 'd'      #option: 'c' ,'d', seperate by c1,c2,c3,c4

if excluded:
    data = np.load("DataSet/regression-data-{}-exclude{}/data.npz".format(model, choose))
    Xtr,ytr = data['x'], data['y']
    data_test = np.load("DataSet/regression-data-{}-exclude{}/data_test.npz".format(model, choose))
    Xva,yva = data_test['x'], data_test['y']
else:
    data = np.load("DataSet/regression-data-{}/data.npz".format(model))
    Xtr,ytr = data['x'], data['y']
    num_data = (Xtr.shape[0]*7)//10
    combined = list(zip(Xtr,ytr))
    random.shuffle(combined)
    Xtr, ytr = zip(*combined)
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    Xtr, Xva, ytr, yva = Xtr[:num_data], Xtr[num_data:], ytr[:num_data], ytr[num_data:]


# 3) Tensors
X_tr_t = torch.from_numpy(Xtr)
y_tr_t = torch.from_numpy(ytr)
X_te_t = torch.from_numpy(Xva)
y_te_t = torch.from_numpy(yva)
torch.set_default_dtype(torch.float64)

# 4) Model (tiny FNN)
d_in  = Xtr.shape[1]
d_out = ytr.shape[1]
model = nn.Sequential(
    nn.Linear(d_in, 16), nn.ReLU(),
    nn.Dropout(0.9),
    nn.Linear(16, 4), nn.ReLU(),
    nn.Dropout(0.9),
    nn.Linear(4, d_out)
)
opt = torch.optim.Adam(model.parameters(), lr=1e-1)
loss_fn = nn.MSELoss()

# 5) Train (full-batch)
model.train()
for epoch in range(20):
    opt.zero_grad()
    pred = model(X_tr_t)
    loss = loss_fn(pred, y_tr_t)
    loss.backward()
    opt.step()
    # if (epoch + 1) % 50 == 0:
    # if True:
        # print(f"Epoch {epoch+1}: train MSE {loss.item():.4f}")

# 6) Test evaluation
model.eval()
with torch.no_grad():
    y_pred_te = model(X_te_t).numpy()

mae  = np.mean(np.abs(y_pred_te - yva),axis=0)
mse  = np.mean((y_pred_te - yva) ** 2,axis=0)
print(mae,mse)
# print(f"\nTest MAE: {mae:.4f}   Test MSE: {mse:.4f}")
# print("Pred shape:", y_pred_te.shape)  # (n_test, n_targets)