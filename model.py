import numpy as np
import pandas as pd
# TORCH:
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Klassen

# Daten importieren
class TrainingsDataset(Dataset):
    def __init__(self,daten):
        
        self.data = np.load(f"data/{daten}")
        self.x = torch.from_numpy(self.data["x"]).float()
        self.y = torch.from_numpy(self.data["y"]).float()
    
        if len(self.x.shape) == 3:
            self.x=self.x.unsqueeze(1)

    def __len__(self): return len(self.x)
    def __getitem__(self, index): return self.x[index],self.y[index]
# Datenformat anpassen damit es mit Modell übereinstimmt.



# Model 

# inspired by: Attosecond Streaking Phase Retrieval Via Deep Learning Methods (page 7, CNN)
class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # Allgemein
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.lin = nn.Linear("Hier Dimension von output_flat einfügen","Output Size überlegen")

        # Block 1
        # HINWEIS: Generierte Daten: 30 Spektrogramme, 800 E_kin, 40 \tau
        self.conv1 = nn.Conv2d(1,20,(21,8))
        self.maxpool1 = nn.MaxPool2d((13,5),stride=3)

        # Block 2
        self.conv2 = nn.Conv2d(20,40,(13,5))
        self.maxpool2 = nn.MaxPool2d((9,3),stride=2)

        # Block 3
        self.conv3 = nn.Conv2d(40,40,(3,2))  # Kern eigentlich 3, hier auf 2 wegen mangelder Breite der Daten
        self.maxpool3 = nn.MaxPool2d((2,2),stride=2)

        # Final
        self.lin2 = nn.Linear("siehe Unten",4)

        # Stand 20260107: start_layer=800*40*30=960000=9.6*1e5

    def forward(self,x):
        # Reshape
        x.view(-1,1,800,40)

        # Block 1
        input_max1 = self.relu(self.conv1(x))
        output1 = self.maxpool1(input_max1)

        # Block 2
        input_max2 = self.relu(self.conv2(output1))
        output2 = self.maxpool2(input_max2)

        # Block 3
        input_max3 = self.relu(self.conv3(output2))
        output3 = self.maxpool3(input_max3)
    
        # flatten
        output_flat=torch.flatten(output3,start_dim=1) # Dimension Error, da zuwenig \tau (Width) Werte. Vor Anwendung Daten/Kernels anpassen.

        # Linear
        input_lin = self.lin(output_flat)
        output_lin = self.relu(input_lin)

        # Dropout
        output_drop = self.dropout(output_lin)

        # final predict
        return self.lin2(output_drop)
    
# Daten importieren

# bis Dato nur Blaupause aus Test Projekt
    batchsize=20
    data = TrainingsDataset("data/training_data.npz")
    dataloader = DataLoader(data,batch_size=batchsize,shuffle=True)
    x,y=next(iter(dataloader))

# Netztrainieren

    model = Model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

    N = int(1e3)

    number_batches = data.__len__()//batchsize+1
    print(f"Number of Batches: {number_batches}")

    best_loss = 10000000000
    for epoch in range(N):
        if epoch%100==0:
            printer=True
        else:
            printer=False
        loss_tracker = np.zeros(number_batches)
        i = -1
        for x_batch,y_batch in dataloader:
            i += 1
            y_predcit = model.forward(x_batch)
            loss = criterion(y_predcit,y_batch)
            loss_tracker[i]=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_mean = loss_tracker.mean()
        if loss_mean < best_loss:
            best_loss = loss_mean
            torch.save(model.state_dict(),"model.pth")
            print(f"Neuer Bestwert: {best_loss} - Gespeichert in Epoche: {epoch}")
        if printer==True:
            print(loss_mean)